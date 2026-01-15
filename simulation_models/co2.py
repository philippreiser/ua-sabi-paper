from typing import Tuple, Dict, Any, List
import numpy as np
from scipy.stats import qmc
import pandas as pd
from scipy.stats import beta

from utils import scale_to_1, scale_from_1
import bayesflow as bf


class CO2:
    def __init__(self, config: Any):
        """
        Initialize the CO2 model with given parameters.

        Args:
            config (Any): Dictionary containing configuration settings, including paths and dataset details.
        """

        self._config = config
        self._sigmaI = config.co2.sigmaI
        self.dim_in = 0
        self.dim_param = 3

        # Constants
        sday = 24 * 3600
        drate = 1600
        self.qco2 = drate / sday
        Bar2Pa = 1e5
        Pmax = 320 * Bar2Pa
        K = 2e-14
        lambda_ = 1e4
        rb = 500
        self.yScale = (Pmax - Bar2Pa * 300) / \
            (self.qco2 * np.log(rb)) * (lambda_ * K)

        # Calculate mc train test splits
        indices = np.arange(0, 10000)
        train_size = 9600
        val_size = 200
        test_size = config.out.n_datasets
        self.train_indices = np.random.choice(
            indices, size=train_size, replace=False)
        remaining_indices = np.setdiff1d(indices, self.train_indices)
        self.val_indices = np.random.choice(
            remaining_indices, size=val_size, replace=False)
        remaining_indices = np.setdiff1d(remaining_indices, self.val_indices)
        self.test_indices = np.random.choice(
            remaining_indices, size=test_size, replace=False)
        self.locations = np.array(self._config.co2.locations)
        self.times = np.array(self._config.co2.times)
        # Check if 'location' is within bounds
        if (self.locations < 1).any() or (self.locations > 250).any():
            raise ValueError(
                f"Invalid 'locations': {self.locations}. Must be between 1 and 250.")

    def load_co2_data(self, path_resp: str, path_input: str) -> pd.DataFrame:
        """
        Load CO2 data from the specified response and input files.

        Args:
            path_resp (str): Path to the response data file.
            path_input (str): Path to the input data file.

        Returns:
            pd.DataFrame: DataFrame containing the combined training and response data.
        """

        # Read response data
        response = pd.read_csv(path_resp, sep='\s+', header=None)
        response.columns = [f"y{j+1}" for j in range(response.shape[1])]
        # Read training data
        training = pd.read_csv(path_input, sep='\s+', header=None)
        # Rename columns (Drop V2, rename V1 -> x1, V3 -> x2, V4 -> x3)
        training = training.drop(columns=[1])
        training = training.rename(columns={0: "x1", 2: "x2", 3: "x3"})
        # Combine training and response data
        co2_data = pd.concat([training, response], axis=1)
        return co2_data

    def load_co2_data_time_location(self, path_input: str, path_resp_template: str, times: List[int]) -> pd.DataFrame:
        """
        Load CO2 data for a specific time and location.

        Args:
            path_input (str): Path to the input data file.
            path_resp_template (str): Template for the response data file paths.
            times (List[int]): List of time points to load data for.

        Returns:
            pd.DataFrame: DataFrame containing the combined training and response data.
        """

        # Load inputs
        training = pd.read_csv(path_input, sep='\s+', header=None)
        training = training.drop(columns=[1])
        training = training.rename(columns={0: "x1", 2: "x2", 3: "x3"})

        # Load responses for each time and location
        response_frames = []
        for time in times:
            time_str = f"{time:04d}"
            path_resp = path_resp_template.format(time=time_str)

            response = pd.read_csv(path_resp, sep='\s+', header=None)
            response.columns = [
                f"y{j+1}_{time}" for j in range(response.shape[1])]
            response_frames.append(response)

        all_responses = pd.concat(response_frames, axis=1)

        # Combine inputs and responses
        co2_data = pd.concat([training, all_responses], axis=1)

        return co2_data

    def load_sobol_data(self, d: int = 2, location: int = 30, time: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads CO2 input and output data for a given time and location.
        The data was generated using Sobol sequences for a specified 'd'. 

        Args:
            d (int, optional): Used to determine the number of model evaluations (default is 2).
            location (int, optional): The location index for which to load the response data (default is 30).
            time (int, optional): The time index for which to load the response data (default is 10).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
                - 'x1' (np.ndarray): injection_rate.
                - 'x2' (np.ndarray): permeability.
                - 'x3' (np.ndarray): porosity.
                - 'response' (np.ndarray): CO2 response data.
        """

        mc = (d+1)**3

        # Load aPC integration points & evaluations
        # Construct file paths
        path_resp = f"data/CO2_Response/d{d}/Tad_Fkt5023_MR0_P0_MC{mc}_N250_T8640000.000000_tP-1.000000_tMR-1.000000_alpha0.500000__S_{time:04d}.dat"
        path_input = f"data/CO2_Response/d{d}/IntegrationPoints.txt"

        co2_data = self.load_co2_data(path_resp, path_input)

        return np.array(co2_data['x1']), np.array(co2_data['x2']), np.array(co2_data['x3']), np.array(co2_data[f"y{location}"])

    def load_abi_data(self, indices=None, split="train", surrogate=None, type="mc") -> Tuple[Dict, Dict, np.ndarray]:
        """
        Loads CO2 input and output data for a given time and location.
        The data was generated either using Monte Carlo evaluation of the prior or Sobol sequences.

        Args:
            indices (np.ndarray, optional): Indices of the datasets to load (default is None).
            split (str, optional): Split type for loading data. Options are "train", "val", "test", or "all" (default is "train").
            surrogate (type, optional): If not none, surrogate model is used for evaluation (default is None).
            type (str, optional): Type of data to load. Options are "mc" or "sobol" (default is "mc").

        Returns:
            Tuple[Dict, Dict, np.ndarray]:
                - out_datasets (Dict): Dictionary containing the input parameters and observations.
                - out_params (Dict): Dictionary containing the input parameters.
                - out_observations (np.ndarray): Noisy observations.
        """

        # Construct file paths
        if type == "mc":
            path_input = f"data/CO2_Response/InputParameters.txt"
            path_resp_template = "data/CO2_Response/samples_10k__S_{time}.dat"
        elif type == "sobol":
            d = self._config.bayesian_apc.polynomial_degree
            mc = (d+1)**3
            path_resp_template = f"data/CO2_Response/d{d}/Tad_Fkt5023_MR0_P0_MC{mc}" + \
                "_N250_T8640000.000000_tP-1.000000_tMR-1.000000_alpha0.500000__S_{time}.dat"
            path_input = f"data/CO2_Response/d{d}/IntegrationPoints.txt"
        time_points = list(range(11))

        co2_data = self.load_co2_data_time_location(
            path_input, path_resp_template, time_points)

        if indices is None:
            if split == "random":
                indices = np.random.choice(
                    co2_data.index, size=self._config.out.n_datasets, replace=False)
            elif split == "all":
                indices = co2_data.index
            elif split == "train":
                indices = self.train_indices
            elif split == "val":
                indices = self.val_indices
            elif split == "test":
                indices = self.test_indices

        n_datasets = len(indices)
        co2_data = co2_data.iloc[indices]

        out_params = {
            "injection_rate": np.array(co2_data["x1"]).reshape(-1, 1),
            "permeability": np.array(co2_data["x2"]).reshape(-1, 1),
            "porosity": np.array(co2_data["x3"]).reshape(-1, 1)
        }

        response_sel = []
        for t in self.times:
            if surrogate is None:
                responses_t = np.array(
                    [co2_data[f"y{loc}_{t}"].values for loc in self.locations]).T
            else:
                responses_t = np.array([surrogate.evaluate(
                    np.array(co2_data["x1"]), np.array(
                        co2_data["x2"]), np.array(co2_data["x3"]),
                    location=location, time=t) for location in self.locations]).T
            response_sel.append(responses_t)

        # Flatten across time × location
        response_sel = np.concatenate(response_sel, axis=1)
        response_noisy = np.random.normal(response_sel, self._sigmaI)
        out_observations = response_noisy

        out_datatsets = {
            'injection_rate': out_params['injection_rate'],
            'permeability': out_params['permeability'],
            'porosity': out_params['porosity'],
            'observations': out_observations
        }

        return out_datatsets, out_params, out_observations

    def load_apc_idxs(self, d: int = 2, location: int = 30, time=10) -> np.ndarray:
        """
        Loads the variable selection indices for a given time and location.
        The data was generated using aPC for a specified 'd'.
        Args:
            d (int, optional): Used to determine the number of model evaluations (default is 2).
            location (int, optional): The location index for which to load the response data (default is 30).
            time (int, optional): The time index for which to load the response data (default is 10).

        Returns:
            np.ndarray: Array of variable selection indices.
        """

        path_idxs = f"data/CO2_Response/apc_idxs/idx_d{d}_co{location}_t{time}varsel_25.csv"
        df = pd.read_csv(path_idxs)
        # adjust for 0-based indexing
        idx_varsel_25 = df.iloc[:, 0].to_numpy() - 1

        return idx_varsel_25

    def evaluate(self):
        """        
        """
        pass

    def scale_to_model_input(self, injection_rate: np.ndarray, permeability: np.ndarray, porosity: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Scales the parameters from physical space to model input space.
        Handles shapes (N, 1) and (N, M, 1).

        Args:
            injection_rate (np.ndarray): Injection rate in physical space.
            permeability (np.ndarray): Permeability in physical space.
            porosity (np.ndarray): Porosity in physical space.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Scaled injection rate, permeability, and porosity.
        """
        injection_rate_scaled = (
            injection_rate / (self.yScale * self.qco2) - 1)
        permeability_scaled = permeability
        porosity_scaled = (porosity - 0.1) / 0.2

        return (injection_rate_scaled, permeability_scaled, porosity_scaled)

    def scale_from_model_input(self, injection_rate: np.ndarray, permeability: np.ndarray, porosity: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Scales the parameters from model input space to physical space.
        Handles shapes (N, 1) and (N, M, 1).

        Args:
            injection_rate (np.ndarray): Injection rate in model input space.
            permeability (np.ndarray): Permeability in model input space.
            porosity (np.ndarray): Porosity in model input space.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Scaled injection rate, permeability, and porosity.
        """
        injection_rate_scaled = (injection_rate + 1) * \
            (self.yScale * self.qco2) * 10000
        permeability_scaled = permeability * 2 + 2
        porosity_scaled = porosity * 0.2 + 0.1

        return (injection_rate_scaled, permeability_scaled, porosity_scaled)

    def scale_dict_from_model_input(self, data: Dict) -> Dict:
        """
        Transforms a parameter dictionary (either out_datasets or draws)
        from model input space to physical space.
        Handles shapes (N, 1) and (N, M, 1).

        Args:
            data (Dict): Dictionary containing the parameters to be transformed.

        Returns:
            Dict: Dictionary containing the transformed parameters.
        """

        def reshape_and_scale(x):
            shape = x.shape
            x = np.squeeze(x)
            flat = x.ravel()
            return flat, shape

        inj_flat, inj_shape = reshape_and_scale(data["injection_rate"])
        perm_flat, perm_shape = reshape_and_scale(data["permeability"])
        poro_flat, poro_shape = reshape_and_scale(data["porosity"])

        inj_phys, perm_phys, poro_phys = self.scale_from_model_input(
            inj_flat, perm_flat, poro_flat)

        return_dict = {
            "injection_rate": inj_phys.reshape(*inj_shape),
            "permeability": perm_phys.reshape(*perm_shape),
            "porosity": poro_phys.reshape(*poro_shape),
        }

        # Include observations if present
        if "observations" in data:
            return_dict["observations"] = data["observations"]

        return return_dict

    def prior(self) -> Dict:
        """
        Generate a random sample from the prior distribution of the CO2 model parameters.

        Returns:
            Dict: Prior distribution of the CO2 model parameters.
        """
        injection_rate = self.yScale * self.qco2 * (1 + beta.rvs(4, 2))
        saturation = 0.05 + 0.2 * beta.rvs(2, 2)
        permeability = beta.rvs(1.25, 1.25)
        porosity = beta.rvs(2.4, 9)

        injection_rate, permeability, porosity = self.scale_to_model_input(
            injection_rate, permeability, porosity)

        return {"injection_rate": injection_rate,
                "permeability": permeability,
                "porosity": porosity}

    def get_simulator(self, surrogate: type = None) -> bf.simulators.Simulator:
        """
        Get the simulator for the CO2 function.

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

        def observation_model(injection_rate: np.ndarray, permeability: np.ndarray, porosity: np.ndarray, n_obs: int) -> Dict:
            """
            Evaluate the CO2 observation model at a given parameter value.

            Args:
                injection_rate (np.ndarray): Parameter value 1.
                permeability (np.ndarray): Parameter value 2.
                porosity (np.ndarray): Parameter value 3.
                n_obs(int): Number of observations.

            Returns:
                Dict: Dictionary containing the (inputs and) model output.
            """

            if surrogate is None:
                raise NotImplementedError
            else:
                injection_rate = np.array([injection_rate])
                porosity = np.array([porosity])
                permeability = np.array([permeability])
                out = [surrogate.evaluate(
                    injection_rate, permeability, porosity,
                    location=location)
                    for location in self.locations]
                out = np.array(out).reshape(-1)
                observations = np.random.normal(
                    out,
                    self._sigmaI, size=n_obs)

            return dict(observations=observations)

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

        out_params = {
            "injection_rate": out_datasets["injection_rate"],
            "permeability": out_datasets["permeability"],
            "porosity": out_datasets["porosity"]}
        out_observations = out_datasets["observations"]

        return out_datasets, out_params, out_observations

    def get_param_names(self) -> List:
        """
        Get the names of the parameters of the CO2 model.

        Returns:
            List: Names of the parameters of the CO2 model.
        """

        return ["injection_rate", "permeability", "porosity"]

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
