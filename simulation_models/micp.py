from typing import Tuple, Dict, Any, List
import numpy as np
from scipy.stats import qmc
import pandas as pd
from scipy.stats import beta
from scipy.io import loadmat
import os

from utils import scale_to_1, scale_from_1
import bayesflow as bf


class MICP:
    def __init__(self, config: Any):
        """
        Initialize the MICP model with given parameters.

        Args:
            config (Any): Dictionary containing configuration settings, including paths and dataset details.
        """

        self._config = config
        self.dim_in = 0
        self.dim_param = 4
        self.locations = np.array(config.micp.locations)
        self.times = np.array(config.micp.times)
        self.data_path = getattr(self._config.micp, "data_path", "data/MICP/full_complexity_model")
        # Constants
        self.sigmaI_measurements = loadmat(os.path.join(self.data_path, 'StdCt.mat'))['StdCt'].flatten()
        measurment_locations = np.array([3, 9, 15, 21, 27, 33, 39, 45])
        self.location_idx = np.where(self.locations == measurment_locations)[0]
        self._sigmaI = self.sigmaI_measurements[self.location_idx]
        self.prior_param_4_upper = config.micp.prior_param_4_upper
        
        # surrogate train test splits
        self.surr_train_indices = np.arange(0, 25)
        self.surr_val_indices = np.arange(0, 25)

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

    def load_sobol_data(self, d: int = 2, location: int = 3, time: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads MICP input and output data for a given time and location.
        The data was generated using polynomial roots and active learning.

        Args:
            d (int, optional): Used to determine the number of model evaluations (default is 2).
            location (int, optional): The location index for which to load the response data (default is 3).
            time (int, optional): The time index for which to load the response data (default is 1).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
                - (np.ndarray): ca1.
                - (np.ndarray): ca2.
                - (np.ndarray): rhoBiofilm.
                - (np.ndarray): kub.
                - (np.ndarray): calcite response data.
        """

        # Load poly roots + active learning points for aPC
        output_ct = loadmat(os.path.join(self.data_path, 'OutputCt.mat'))['OutputCt'][self.surr_train_indices, :]
        collocation_points_base = loadmat(os.path.join(self.data_path, 'CollocationPointsBase.mat'))['CollocationPointsBase'][self.surr_train_indices, :]

        return collocation_points_base[:, 0], collocation_points_base[:, 1], collocation_points_base[:, 2], collocation_points_base[:, 3], output_ct[:, location]

    def generate_mc_samples(self, n_samples: int = 10000) -> np.ndarray:
        """
        Generate Monte Carlo samples for the MICP model parameters.

        Args:
            n_samples (int, optional): Number of samples to generate (default is 10000).

        Returns:
            np.ndarray: shape (n_samples, 4) array containing the generated samples.
                The columns represent:
                - ca1
                - ca2
                - rhoBiofilm
                - kub
        """
        np.random.seed(0)
        mc_points = np.zeros((n_samples, self.dim_param))
        param_names = self.get_param_names()
        param_bounds = self.get_param_bounds()

        for j, key in enumerate(param_names):
            lower, upper = param_bounds[key]
            mc_points[:, j] = np.random.uniform(low=lower, high=upper, size=n_samples)
        return mc_points

    def load_abi_data(self, indices=None, split="train", surrogate=None, type="mc") -> Tuple[Dict, Dict, np.ndarray]:
        """
        Loads MICP input and output data for a given time and location.
        The data was generated either using Monte Carlo evaluation of the prior or Sobol sequences.

        Args:
            surrogate (type, optional): If not none, surrogate model is used for evaluation (default is None).
            type (str, optional): Type of data to load. Options are "mc" or "sobol" (default is "mc").

        Returns:
            Tuple[Dict, Dict, np.ndarray]:
                - out_datasets (Dict): Dictionary containing the input parameters and observations.
                - out_params (Dict): Dictionary containing the input parameters.
                - out_observations (np.ndarray): Noisy observations.
        """

        # load MICP data
        if surrogate is None:
            # Load data from the full complexity model
            if split == "train" or split == "all":
                indices = self.surr_train_indices
            elif split == "val":
                indices = self.surr_val_indices
            elif split == "test":
                indices = self.surr_val_indices
            output_ct = loadmat(os.path.join(self.data_path, 'OutputCt.mat'))['OutputCt'][indices, :]
            param_samples = loadmat(os.path.join(self.data_path, 'CollocationPointsBase.mat'))['CollocationPointsBase'][indices, :]
        else:
            # Load MC-sampled params
            if split == "train":
                indices = self.train_indices
            elif split == "val":
                indices = self.val_indices
            elif split == "test":
                indices = self.test_indices
            output_ct = None
            param_samples = self.generate_mc_samples()[indices, :]

        # Create params dict
        param_names = self.get_param_names()
        out_params = {
            name: param_samples[:, i].reshape(-1, 1)
            for i, name in enumerate(param_names)
        }

        # Create observations either directly from output_ct or via surrogate
        if surrogate is None:
            if output_ct is None:
                raise ValueError("No output data available and no surrogate provided.")
            out_observations = output_ct[:, self.locations]  # shape (n_samples, n_locations)
        else:
            # Evaluate surrogate at each location
            n_samples = param_samples.shape[0]
            random_idx = np.random.randint(
                low=0, high=surrogate.n_total_samples, size=n_samples)
            params = [param_samples[:, i] for i in range(self.dim_param)]
            out_observations = np.array([
                surrogate.evaluate(*params, location=loc, time=self.times[0], random_idx=random_idx)
                for loc in self.locations
            ]).T  # shape (n_samples, n_locations)
        out_observations = np.random.normal(out_observations, self._sigmaI)

        # Final dataset dictionary
        out_datasets = {**out_params, 'observations': out_observations}

        return out_datasets, out_params, out_observations
    
    def load_ct_measurements(self): 
        """
        Load the CT measurements from the MICP dataset.

        Returns:
            dict: out_dataset with CT measurements.
        """
        ct_measurements = loadmat(os.path.join(self.data_path, 'observedCt.mat'))['observedCt'][:, self.location_idx]
        out_dataset = {
            "observations": ct_measurements.reshape(1, -1)
        }
        return out_dataset


    def evaluate(self):
        """
        Online training not feasible for expensive MICP model.
        """
        pass

    def get_param_names(self) -> List:
        """
        Get the names of the parameters of the MICP model.

        Returns:
            List: Names of the parameters of the MICP model.
        """

        # return ["ca1", "ca2", "rhoBiofilm", "kub"]
        return [r"$c_{a,1}$", r"$c_{a,2}$", r"$\rho_f$", r"$k_{ub}$"]
    
    def get_param_bounds(self) -> Dict:
        """
        Get the bounds of the parameters of the MICP model.

        Returns:
            Dict: Bounds of the parameters of the MICP model.
        """
        param_names = self.get_param_names()
        param_bounds = {
            param_names[0]: (1e-10, 1e-7),
            param_names[1]: (1e-10, 1e-6),
            param_names[2]: (1.0, 15.0),
            param_names[3]: (1e-5, self.prior_param_4_upper) # originally (1e-5, 5e-4)
        }
        return param_bounds
    
    def rename_parameter_keys(self, draws: dict) -> dict:
        """
        Rename parameter keys from old convention (param_1, param_2, …)
        to new LaTeX-style names from model.get_param_names().

        Args:
            draws (dict): Dictionary of arrays containing parameter draws and possibly 
            additional entries. Keys may follow the old convention 
            (``"param_i"``) or already match the model parameter names.

        Returns:
            Dict: A dictionary with parameter keys renamed to the model-defined names. 
            Non-parameter keys are preserved.
        """
        param_names = self.get_param_names()
        
        renamed = {}
        old_keys = []
        for i, pname in enumerate(param_names, start=1):
            old_key = f"param_{i}"
            if old_key in draws:
                renamed[pname] = draws[old_key]
                old_keys.append(old_key)
            elif pname in draws:  # already correct name
                renamed[pname] = draws[pname]
            else:
                raise KeyError(f"Neither {old_key} nor {pname} found in draws.")
        
        # also keep non-parameter keys (like "observations")
        for k, v in draws.items():
            if k not in renamed and k not in old_keys:
                renamed[k] = v
        
        return renamed

