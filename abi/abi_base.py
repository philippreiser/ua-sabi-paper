from typing import Any, Dict, Type, Tuple
import os
import keras
import dill
import numpy as np

import bayesflow as bf

import importlib

import diagnoser as d
importlib.reload(d)


class ABI:
    """
    Perform Amortized Bayesian inference.
    """

    def __init__(self, config: Any, model: Type, surrogate: Type) -> None:
        """
        Initialize ABI.

        Args:
            config (Any): Dictionary containing configuration settings, including paths and dataset details.
            model (Type): True model.
            surrogate (Type): Surrogate model.
        """

        self._config = config
        self._model = model
        self._surrogate = surrogate

    def meta(self, batch_size: int) -> Dict:
        """
        Meta function needed number of observations.

        Args:
            batch_size (int): Batch size.

        Returns:
            Dict: Number of observations.
        """
        if self._config.abi.n_obs[0] != self._config.abi.n_obs[1]:
            n_obs = np.random.randint(
                self._config.abi.n_obs[0], self._config.abi.n_obs[1])
        else:
            n_obs = self._config.abi.n_obs[0]
        return dict(n_obs=n_obs)

    def define_prior_function(self) -> Type:
        """
        Define the prior function as bayesflow needs functions and not class methods.

        Returns:
            Type: Prior distribution function.
        """

        def prior() -> Dict:
            """
            Return the model specific prior distribution.

            Returns:
                Dict: Prior distribution.
            """

            prior = self._model.prior()

            return prior

        return prior

    def define_observation_model_function(self) -> Type:
        """
        Define the observation model function as bayesflow needs functions and not class methods.

        Returns:
            observation_model (Type): Observation model function.
        """

        raise NotImplementedError("Must be implemented by specific subclass.")

    def define_adapter(self) -> Type:
        """
        Define adapter object.

        Returns:
            adapter (Type): Adapter object.
        """

        raise NotImplementedError("Must be implemented by specific subclass.")

    def _build_approximator(self, adapter: bf.adapters.Adapter) -> bf.ContinuousApproximator:
        """ 
        Build the approximator.

        Args:
            adapter (bf.adapters.Adapter): Adapter object.
        Returns:
            bf.ContinuousApproximator: Approximator object.
        """

        sumnet_type = getattr(self._config.abi, "sumnet", "deep_set")
        if sumnet_type == "deep_set":
            summary_net = bf.networks.DeepSet(
                depth=self._config.abi.sumnet_depth,
                summary_dim=self._config.abi.sumnet_dim,
                dropout=getattr(self._config.abi, "sumnet_dropout", 0.05),
                mlp_widths_invariant_outer=(64, 4)
                )
        elif sumnet_type == "time_series_network":
            summary_net = bf.networks.TimeSeriesNetwork(
                dropout=getattr(self._config.abi, "sumnet_dropout", 0.05)
                )
        elif sumnet_type == "none":
            summary_net = None

        if self._config.abi.infnet == "flowmatching":
            inference_net = bf.networks.FlowMatching()
        elif self._config.abi.infnet == "couplingflow":
            inference_net = bf.networks.CouplingFlow()
        elif self._config.abi.infnet == "spline_couplingflow":
            inference_net = bf.networks.CouplingFlow(transform="spline")
        else:
            raise ValueError("Unsupported inference network")
        
        try:
            approximator = bf.ContinuousApproximator(
                summary_network=summary_net,
                inference_network=inference_net,
                adapter=adapter,
                standardize="all"
            )
        except TypeError:
            approximator = bf.ContinuousApproximator(
                summary_network=summary_net,
                inference_network=inference_net,
                adapter=adapter
            )

        return approximator

    def _build_optimizer(self, dataset: Type) -> keras.optimizers.Optimizer:
        """
        Build the optimizer.

        Args:
            dataset (Type): Dataset object.
        Returns:
            keras.optimizers.Optimizer: Optimizer object.
        """
        sched = self._config.abi.learning_rate_scheduler
        base_lr = self._config.abi.learning_rate
        decay_steps = self._config.abi.epochs * dataset.num_batches
        if sched == "fixed":
            lr = base_lr
        elif sched == "cosine":
            lr = keras.optimizers.schedules.CosineDecay(
                base_lr, decay_steps=decay_steps, alpha=self._config.abi.learning_rate_hp)
        elif sched == "exponential":
            lr = keras.optimizers.schedules.ExponentialDecay(
                base_lr, decay_steps=decay_steps, decay_rate=self._config.abi.learning_rate_hp)
        else:
            raise ValueError("Learning rate schedule not supported.")

        opt_type = self._config.abi.optimizer
        if opt_type == "adam":
            return keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
        elif opt_type == "adamw":
            return keras.optimizers.AdamW(learning_rate=lr, clipnorm=1.0)
        elif opt_type == "nadam":
            return keras.optimizers.Nadam(learning_rate=lr, clipnorm=1.0)
        else:
            raise ValueError("Optimizer not supported.")

    def _train(self, approximator: Type, dataset: Type, val_dataset: Type) -> Type:
        """
        Train the ABI model.

        Args:
            approximator (Type): Approximator object.
            dataset (Type): Dataset object.
            val_dataset (Type): Validation dataset object.
        Returns:
            Type: Training history.
        """
        optimizer = self._build_optimizer(dataset)
        approximator.compile(optimizer=optimizer)

        diagnoser = d.Diagnoser(self._config)
        history = diagnoser.measure_execution_time(
            "abi_training_time",
            approximator.fit,
            epochs=self._config.abi.epochs,
            dataset=dataset,
            validation_data=val_dataset)
        return history

    def train(self) -> Tuple[bf.simulators.Simulator, bf.approximators.Approximator, Dict]:
        """
        Train the ABI model using simulated data generated on-the-fly.

        This method constructs a 'bf.Simulator' object based on the configured prior and observation model.
        Training data is then generated from this simulator in one of two modes:

        - **Online (full-budget)**: If 'config.abi.low_budget' is 'False', an 'OnlineDataset' is created 
        which generates data batches during training.

        - **Offline (low-budget)**: If 'low_budget' is 'True', a fixed number of training points is sampled 
        from the simulator to create an 'OfflineDataset'. The number of training points is aligned with the 
        surrogate model budget ('bayesian_pce.log2_n_training_points'). A small validation dataset is also 
        sampled from the simulator.

        Returns:
            simulator (bf.simulators.Simulator): The constructed simulator used for data generation.
            approximator (bf.approximators.Approximator): The ABI approximator model.
            history (Dict): A dictionary containing training diagnostics and metrics.
        """

        prior = self.define_prior_function()
        observation_model = self.define_observation_model_function()
        simulator = bf.make_simulator(
            [prior, observation_model], meta_fn=self.meta)

        adapter = self.define_adapter()
        approximator = self._build_approximator(adapter)

        if not self._config.abi.low_budget:
            dataset = bf.OnlineDataset(
                simulator, batch_size=self._config.abi.batch_size,
                num_batches=self._config.abi.n_batches, adapter=adapter, workers=8)
            val_dataset = None
        else:
            # low budget training: use number of training points used for surrogate training
            n_train = int(
                (2**self._config.bayesian_pce.log2_n_training_points)/self._config.out.n_obs)
            train_data = simulator.sample(n_train)
            dataset = bf.OfflineDataset(
                train_data, batch_size=2**self._config.bayesian_pce.log2_n_training_points, adapter=adapter)
            val_dataset = bf.OfflineDataset(
                simulator.sample(20), batch_size=20, adapter=adapter)

        history = self._train(approximator, dataset, val_dataset)

        self.save(simulator, approximator, history)
        return simulator, approximator, history

    def train_with_presimulated_data(self) -> Tuple[bf.approximators.Approximator, Dict]:
        """
        Train the ABI model using pre-simulated data.

        This method loads fixed training and validation datasets from disk via 'self._model.load_abi_data()'.
        The training budget is controlled via the 'config.abi.low_budget' flag:

        - **Full-budget** ('low_budget = False'): Loads large, pre-generated train/val splits.
        - **Low-budget** ('low_budget = True'): Loads reduced training data generated with a low-discrepancy 
        Sobol sequence (e.g., for use with surrogate models or constrained settings).

        If 'config.surrogate == "true_model"', true model data is used. Otherwise, the output is generated using a surrogate model.
        Validation data may be selected based on whether 'config.out.use_true_model' is set.

        Returns:
            approximator (bf.approximators.Approximator): The ABI approximator model.
            history (Dict): A dictionary containing training diagnostics and metrics.
        """

        adapter = self.define_adapter()
        approximator = self._build_approximator(adapter)

        if self._config.surrogate == "true_model":
            if not self._config.abi.low_budget:
                train = self._model.load_abi_data(split="train")[0]
                val = self._model.load_abi_data(split="val")[0]
            else:
                train = self._model.load_abi_data(split="all", type="sobol")[0]
                val = self._model.load_abi_data(split="val")[0]
        else:
            train = self._model.load_abi_data(
                split="train", surrogate=self._surrogate)[0]
            if self._config.out.use_true_model:
                val = self._model.load_abi_data(split="val")[0]
            else:
                val = self._model.load_abi_data(
                    split="val", surrogate=self._surrogate)[0]
        dataset = bf.OfflineDataset(
            train, batch_size=self._config.abi.batch_size, adapter=adapter)
        val_dataset = bf.OfflineDataset(
            val, batch_size=self._config.abi.batch_size, adapter=adapter)
        history = self._train(approximator, dataset, val_dataset)
        self.save(None, approximator, history)
        return approximator, history

    def save(self, simulator: Type, approximator: Type, history: Type) -> None:
        """
        Save the ABI model.

        Args:
            simulator (Type): Simulator object.
            approximator (Type): Approximator object.
            history (Type): Training history.
        """

        save_path = os.path.join(self._config.output_path, "abi")
        os.makedirs(save_path, exist_ok=True)

        with open(os.path.join(save_path, "simulator.pkl"), "wb") as file:
            dill.dump(simulator, file)

        approximator.save(os.path.join(save_path, "approximator.keras"))

        np.savez(os.path.join(save_path, "losses.npz"), **history.history)

    def load(self, path: str) -> Tuple[Type, Type]:
        """
        Load the ABI model.

        Args:
            path (str): Path to the ABI model.

        Returns:
            Tuple[Type, Type]: Simulator and approximator objects.
        """

        load_path = os.path.join(path, "abi")

        if not os.path.exists(load_path):
            raise FileNotFoundError(
                f"The specified path does not exist: {load_path}")

        with open(os.path.join(load_path, "simulator.pkl"), "rb") as file:
            simulator = dill.load(file)

        @keras.saving.register_keras_serializable(package="custom_package")
        def forward_transform(n):
            return np.sqrt(n)

        @keras.saving.register_keras_serializable(package="custom_package")
        def inverse_transform(n):
            return n**2

        approximator = keras.models.load_model(os.path.join(
            load_path, "approximator.keras"), safe_mode=False)

        return simulator, approximator
