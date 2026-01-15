import os
import json
from datetime import datetime
import shutil
import numpy as np
from typing import Dict
import logging
import glob


class JSONDict:
    """
    Convert dictionaries into objects with attribute-style access for nested dictionaries.

    Attributes can be accessed as attributes instead of key-value pairs, simplifying syntax.
    """

    def __init__(self, data: dict) -> None:
        self.__dict__.update({
            key: JSONDict(value) if isinstance(value, dict) else value
            for key, value in data.items()
        })

    def to_dict(self) -> dict:
        return {key: value.to_dict() if isinstance(value, JSONDict) else value
                for key, value in self.__dict__.items()}

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=4)


def read_config(config_path: str) -> JSONDict:
    """
    Load a JSON configuration file and return as a JSONDict object.

    Args:
        config_path (str): File path to the JSON configuration file.

    Returns:
        JSONDict: Parsed configuration with attribute-style access.
    """

    with open(config_path, "r") as file:
        config = JSONDict(json.load(file))
    assert config.abi.n_obs[0] <= config.out.n_obs <= config.abi.n_obs[1], (
        f"Number of observations ({config.out.n_obs}) must be within the range "
        f"[{config.abi.n_obs[0]}, {config.abi.n_obs[1]}] used for ABI training.")

    return config


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def setup_logger(output_path: str) -> None:
    """
    Sets up the logger with the specified output path.

    Args:
        output_path (str): Path to the output folder.
    """

    global logger

    # Remove any existing handlers to prevent duplicate logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # File handler
    file_handler = logging.FileHandler(os.path.join(output_path, "info.log"))
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def create_output_folders(config_path: str) -> str:
    """
    Create output folders based on current timestamp and dataset structure.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        str: Path to the created root results folder.
    """

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join("results", timestamp)
    os.makedirs(path, exist_ok=True)

    # Copy the configuration file to the output folder
    shutil.copy(config_path, path)

    return path


def save_out(path: str, out_datasets: Dict, out_params: Dict, out_observations: np.ndarray) -> None:
    """
    Save the output datasets, parameters, and observations to the output folder.

    Args:
        path (str): Path to the output folder.
        out_datasets (Dict): Output datasets.
        out_params (Dict): Output parameters.
        out_observations (np.ndarray): Output observations.
    """

    save_path = os.path.join(path, "out")

    os.makedirs(save_path, exist_ok=True)

    np.savez(os.path.join(save_path, "out_datasets.npz"), **out_datasets)
    np.savez(os.path.join(save_path, "out_params.npz"), **out_params)
    np.savez(os.path.join(save_path, "out_observations.npz"), out_observations)


def load_out(path: str) -> Dict:
    """
    Load the output datasets, parameters, and observations from the output folder.

    Args:
        path (str): Path to the output folder.

    Returns:
        Tuple[Dict, Dict, np.ndarray]: Output datasets, parameters, and observations.
    """

    load_path = os.path.join(path, "out")

    if not os.path.exists(load_path):
        raise FileNotFoundError(
            f"The specified path does not exist: {load_path}")

    out_datasets = np.load(os.path.join(load_path, "out_datasets.npz"))
    out_params = np.load(os.path.join(load_path, "out_params.npz"))
    out_observations = np.load(os.path.join(load_path, "out_observations.npz"))

    return dict(out_datasets), dict(out_params), out_observations["arr_0"]


def save_draws(path: str, parameter_draws: np.ndarray, file_name: str="parameter_draws") -> None:
    """
    Save the parameter draws to the output folder.

    Args:
        path (str): Path to the output folder.
        parameter_draws (np.ndarray): Output observations.
    """

    save_path = os.path.join(path, "out")

    os.makedirs(save_path, exist_ok=True)

    np.savez(os.path.join(save_path, f"{file_name}.npz"), **parameter_draws)


def load_draws(path: str) -> np.ndarray:
    """
    Load the parameter draws from the output folder.

    Args:
        path (str): Path to the output folder.

    Returns:
        np.ndarray: The parameter draws.
    """

    load_path = os.path.join(path, "out")

    if not os.path.exists(load_path):
        raise FileNotFoundError(
            f"The specified path does not exist: {load_path}")

    parameter_draws = np.load(os.path.join(load_path, "parameter_draws.npz"))

    return dict(parameter_draws)


def scale_to_1(x: np.ndarray, lb: float, ub: float) -> np.ndarray:
    """
    Scale the elements of an array from [lb, ub] to [-1, 1]

    Args:
        x (np.ndarray): array to scale
        lb (float): lower bound
        ub (float): upper bound

    Returns:
        np.ndarray: array
    """

    x_scaled = 2 / (ub - lb) * (x - lb) - 1

    return x_scaled


def scale_from_1(x: np.ndarray, lb: float, ub: float) -> np.ndarray:
    """
    Scale the elements of an array from [-1, 1] to [lb, ub]

    Args:
        x (np.ndarray): array to scale
        lb (float): lower bound
        ub (float): upper bound

    Returns:
        np.ndarray: scaled array
    """

    x_scaled = (ub - lb) / 2 * (x + 1) + lb

    return x_scaled

def load_parameter_draws_from_runs(base_path: str, model) -> dict:
    """
    Loads parameter draws from multiple run folders under base_path.
    The method name is extracted from the filename: config_<method>.json

    Args:
        base_path (str): Path containing run subfolders.

    Returns:
        dict: Mapping from method name to parameter draw dictionary.
    """
    draws_dict = {}

    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # Find config_<method>.json file
        config_files = glob.glob(os.path.join(folder_path, "config_*.json"))
        if len(config_files) != 1:
            continue  # Skip if config is missing or ambiguous

        config_path = config_files[0]
        method_name = os.path.splitext(os.path.basename(config_path))[0].replace("config_", "")

        draws_path = os.path.join(folder_path, "out", "parameter_draws_real.npz")
        if not os.path.exists(draws_path):
            continue

        # Load draws
        raw_draws = dict(np.load(draws_path))

        raw_draws = model.rename_parameter_keys(raw_draws)

        # Rename param_4 -> r"$k_{ub}$"
        if "param_4" in raw_draws:
            raw_draws[r"$k_{ub}$"] = raw_draws.pop("param_4")

        draws_dict[method_name] = raw_draws

    return draws_dict

