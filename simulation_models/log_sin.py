from typing import Tuple, Dict, Any, List
import numpy as np
from scipy.stats import qmc

from utils import scale_to_1, scale_from_1
import bayesflow as bf


class LogSin:
    def __init__(self, config: Any, w_range: Tuple = (0.6, 1.4), x_range: Tuple = (1, 200)):
        """
        Initialize the LogSin function (toy problem) with given parameters.

        Args:
            config (Any): Dictionary containing configuration settings, including paths and dataset details.
            w_range (Tuple): The range for parameter w (default: (0.6, 1.4)).
            x_range (Tuple): The range for input x (default: (1, 200)).
        """

        self._config = config
        self._w_range = w_range
        self._x_range = x_range
        self._sigmaI = config.log_sin.sigmaI
        self.dim_in = 1
        self.dim_param = 1

    def evaluate(self, w: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the LogSin function at a given parameter w and point x.

        Args:
            w (np.ndarray): Value of w.
            x (np.ndarray): Value of x.

        Returns:
            np.ndarray: The result of the LogSin function at (w, x).
        """

        return w * np.log(x) + 0.01 * x + 1 + np.sin(0.05 * x)

    def random_samples(self, n_samples: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a random sample from the LogSin function's input space.

        Args:
            n_samples (int, optional): Number of random samples to generate (default: 1).

        Returns:
            Tuple: Random sample from the LogSin function's input space.
        """

        w = np.random.uniform(self._w_range[0], self._w_range[1], n_samples)
        x = np.random.uniform(self._x_range[0], self._x_range[1], n_samples)

        val = self.evaluate(w, x)

        return w, x, val

    def sobol_sequence(self, m: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate 2D Sobol' sequence of length 2^m from the LogSin function's 
        input space.

        Args:
            m (int, optional): Logarithm in base 2 of the number of samples; i.e., n = 2^m (default: 1).

        Returns:
            Tuple: 2D Sobol' sequence of length 2^m from the LogSin function's input space.
        """

        sampler = qmc.Sobol(d=2, scramble=False)
        input_sample = sampler.random_base2(m)
        input_sample = qmc.scale(input_sample,
                                 l_bounds=[self._w_range[0], self._x_range[0]],
                                 u_bounds=[self._w_range[1], self._x_range[1]])
        w = input_sample[:, 0]
        x = input_sample[:, 1]

        val = self.evaluate(w, x)

        return w, x, val

    def prior(self) -> Dict:
        """
        Generate a random sample from the prior distribution of the LogSin function's parameter.

        Returns:
            Dict: Prior distribution of the LogSin function's parameter.
        """

        w = np.float32(np.random.normal(1, 0.2))

        return dict(w=w)

    def inputs(self, n_obs: int) -> np.ndarray:
        """
        Generate a random sample from the input space of the LogSin function.

        Args:
            n_obs (int): Number of observations.

        Returns:
            np.ndarray: Distribution of the input space of the LogSin function.
        """

        x = np.float32(np.random.uniform(
            self._x_range[0], self._x_range[1], n_obs))

        return x

    def scale_input_to_1(self, x: np.array) -> np.ndarray:
        """
        Scale x from (self._x_range[0], self._x_range[1]) to (-1, 1)

        Args:
            x (np.ndarray): unscaled x.

        Returns:
            np.ndarray: scaled x.
        """

        x_scaled = scale_to_1(x, self._x_range[0], self._x_range[1])

        return (x_scaled, )

    def scale_input_from_1(self, x: np.array) -> np.ndarray:
        
        """
        Scale x from (-1, 1) to (self._x_range[0], self._x_range[1])

        Args:
            x (np.ndarray): scaled x.

        Returns:
            np.ndarray: unscaled x.
        """

        x_scaled = scale_from_1(x, self._x_range[0], self._x_range[1])

        return (x_scaled, )

    def scale_parameter_to_1(self, w: np.array) -> np.ndarray:
        """
        Scale w from (self._w_range[0], self._w_range[1]) to (-1, 1)

        Args:
            w (np.ndarray): unscaled w.

        Returns:
            np.ndarray: scaled w.
        """

        w_scaled = scale_to_1(w, self._w_range[0], self._w_range[1])

        return (w_scaled, )

    def scale_parameter_from_1(self, w: np.array) -> np.ndarray:
        """
        Scale w from (-1, 1) to (self._w_range[0], self._w_range[1])

        Args:
            w (np.ndarray): scaled w.

        Returns:
            np.ndarray: unscaled w.
        """

        w_scaled = scale_from_1(w, self._w_range[0], self._w_range[1])

        return (w_scaled, )

    def scale_inputs_to_1(self, w: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize w and x to (-1, 1)

        Args:
            w (np.ndarray): Value of w.
            x (np.ndarray): Value of x.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Normalzed inputs.
        """

        w_scaled = self.scale_parameter_to_1(w)[0]
        x_scaled = self.scale_input_to_1(x)[0]

        return w_scaled, x_scaled

    def get_simulator(self, surrogate: type = None) -> bf.simulators.Simulator:
        """
        Get the simulator for the LogSin function.

        Args:
            surrogate (type): Trained surrogate. If not none, it will be used
            as the simulator instead of the true model.

        Returns:
            bf.simulators.Simulator: Simulator object.
        """

        def meta(batch_size: int = 1) -> Dict:
            """
            Meta function needed number of observations.

            Args:
                batch_size (int): Batch size.

            Returns:
                Dict: Number of observations.
            """

            n_obs = self._config.out.n_obs

            return dict(n_obs=n_obs)

        def prior() -> Dict:
            """
            Return the model specific prior distribution.

            Returns:
                Dict: Prior distribution.
            """

            prior = self.prior()

            return prior

        def observation_model(w: np.ndarray, n_obs: int) -> Dict:
            """
            Evaluate the LogSin observation model at a given parameter value.

            Args:
                w(np.ndarray): Parameter value.
                n_obs(int): Number of observations.

            Returns:
                Dict: Dictionary containing the (inputs and) model output.
            """

            inputs = self.inputs(n_obs)

            if surrogate is None:
                observations = np.random.normal(
                    self.evaluate(np.repeat(w, n_obs), inputs), self._sigmaI, size=n_obs)
            else:
                observations = np.random.normal(
                    surrogate.evaluate(np.repeat(w, n_obs), inputs),
                    self._sigmaI, size=n_obs)

            return dict(inputs=inputs, observations=observations)

        simulator = bf.make_simulator([prior, observation_model], meta_fn=meta)
        return simulator

    def get_simulations(self, simulator: bf.simulators.Simulator) -> Tuple[Dict, Dict, np.ndarray]:
        """
        Get simulations from the simulator.

        Args:
            simulator (bf.simulators.Simulator): Simulator object.

        Returns:
            Tuple[Dict, Dict, np.ndarray]: Output datasets, output parameters, and observations.
        """

        out_datasets = simulator.sample(self._config.out.n_datasets)

        out_params = {"w": out_datasets["w"]}
        out_observations = out_datasets["observations"]

        return out_datasets, out_params, out_observations

    def get_param_names(self) -> List:
        """
        Get the names of the parameters of the Logistic function.

        Returns:
            List: Names of the parameters of the Logistic function.
        """

        return ["w"]

    def get_plot_inputs(self, n_params: int = 100, n_inputs: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get inputs to generate predictive plots.

        Args:
            n_params (int, optional): Number of parameter points (default: 100).
            n_inputs (int, optional): Number of input points (default: 100).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Model parameters and inputs.
        """

        w_plot = np.linspace(self._w_range[0], self._w_range[1], n_params)
        x_plot = np.linspace(self._x_range[0], self._x_range[1], n_inputs)

        return w_plot, x_plot
