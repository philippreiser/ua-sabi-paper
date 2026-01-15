from typing import Dict, Any, Type, List
import numpy as np
import time
import os
import json
import bayesflow as bf


class Diagnoser:
    """
    Manages time measurements and calibration metrics of the chosen inference.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initializes the Diagnoser, setting up directories for creating and saving visualizations based on the configuration.

        Args:
            config (Dict[str, Any]): Dictionary containing configuration settings, including paths and dataset details.
        """

        self._config = config

    def measure_execution_time(self, keyword: str, func: Type, *args, **kwargs) -> Any:
        """
        Measures the execution time of a function and saves it to a file.

        Args:
            func (Type): Function to be measured.
            *args: Arguments of the function.
            **kwargs: Keyword arguments of the function.

        Returns:
            Any: Result of the function.
        """

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time

        self.save(keyword, execution_time)

        return result

    def measure_calibration_error(self, target_draws: np.ndarray, estimate_draws: np.ndarray, variable_names: List[str]) -> None:
        """
        Measures the calibration error of the inference and saves it to a file.

        Computes an aggregate score for the marginal calibration error over an ensemble of approximate
        posteriors. The calibration error is given as the aggregate (e.g., median) of the absolute deviation
        between an alpha-CI and the relative number of inliers from estimates over multiple alphas (dep on resolution) in
        (0, 1).

        Args:
            target_draws (np.ndarray): Target draws.
            estimate_draws (np.ndarray): Approximated draws.
        """

        os.makedirs(os.path.join(self._config.output_path,
                    "diagnosis"), exist_ok=True)

        result = bf.diagnostics.metrics.calibration_error(targets=target_draws, estimates=estimate_draws,
                                                          resolution=20,
                                                          aggregation=np.median,
                                                          min_quantile=0.005,
                                                          max_quantile=0.995,
                                                          variable_names=variable_names
                                                          )

        filename = os.path.join(self._config.output_path,
                                "diagnosis", "calibration_error.npz")

        np.savez(filename, **result)

        return result

    def save(self, keyword: str, value: Any) -> None:
        """
        Saves the given keyword and value to a JSON file.

        Args:
            keyword (str): Keyword to be saved.
            value (Any): Value to be saved.
        """

        data = {keyword: value}

        os.makedirs(os.path.join(self._config.output_path,
                    "diagnosis"), exist_ok=True)
        filename = os.path.join(self._config.output_path,
                                "diagnosis", "time.json")

        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                try:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, dict):
                        existing_data = {}
                except json.JSONDecodeError:
                    existing_data = {}
        else:
            existing_data = {}

        existing_data.update(data)

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=4)
    
    def compute_sobol_indices(self, surrogate: Type, param_names: list) -> Dict:
        """
        Compute Sobol indices for a surrogate model.

        Args:
        surrogate (Type): Surrogate model.
        param_names (list): List of parameter names corresponding to the surrogate model.

        Returns:
        Dict: Dictionary containing Sobol indices for each parameter and each coefficient sample per time and location.
        """

        n_params = len(param_names)
        n_times = surrogate.c_samples.shape[0]
        n_locs = surrogate.c_samples.shape[1]
        n_samples = surrogate.c_samples.shape[2]

        sobol_maps = {param: np.zeros((n_times, n_locs, n_samples)) for param in param_names}

        for t in range(n_times):
            for l in range(n_locs):
                for s in range(n_samples):
                    coeffs = surrogate.c_samples[t, l, s, :]
                    alphas = surrogate.poly_idx[1:, :]

                    total_var = np.sum(coeffs**2)
                    if total_var == 0:
                        continue # avoid division by zero

                    for i, param in enumerate(param_names):
                        mask = (alphas[:, i] > 0) & np.all(alphas[:, np.arange(n_params) != i] == 0, axis=1)
                        # mask = (alphas[:, i] > 0))
                        sobol = np.sum(coeffs[mask] ** 2) / total_var
                        sobol_maps[param][t, l, s] = sobol

        return sobol_maps