from typing import Any, Dict, Type
import os
import numpy as np
import keras
import bayesflow as bf

from .abi_base import ABI
from surrogate_methods.bayesian_pce import BayesianPCEBase
from surrogate_methods.bayesian_apc import BayesianaPCBase


class ABI_CO2(ABI):
    """
    Perform Amortized Bayesian inference.
    """

    def __init__(self, config: Any, model: Type, surrogate: Type) -> None:
        """
        Initialize ABI for CO2.

        Args:
            config (Dict): Dictionary containing configuration settings, including paths and datasets details.
            model (Type): True model.
            surrogate (Type): Surrogate model.
        """

        super().__init__(config, model, surrogate)

    def define_observation_model_function(self) -> Type:
        """
        Define the observation model function as bayesflow needs functions and not class methods.

        Returns:
            observation_model (Type): Observation model function.
        """

        def observation_model(injection_rate: np.ndarray, permeability: np.ndarray, porosity: np.ndarray, n_obs: int) -> Dict:
            """
            Evaluate the CO2 observation model at a given parameter value.

            Args:
                x1 (np.ndarray): Parameter 1 value.
                x2 (np.ndarray): Parameter 2 value.
                x3 (np.ndarray): Parameter 3 value.
                n_obs (int): Number of observations.

            Returns:
                Dict: Dictionary containing the (inputs and) model output.
            """

            if isinstance(self._surrogate, (BayesianPCEBase, BayesianaPCBase)):
                injection_rate = np.array([injection_rate])
                porosity = np.array([porosity])
                permeability = np.array([permeability])
                out = [[self._surrogate.evaluate(injection_rate, permeability, porosity, location=location, time=time)
                       for location in self._model.locations] for time in self._model.times]
                out = np.array(out).reshape(-1)
            else:
                raise NotImplementedError
            observations = np.random.normal(
                out, self._model._sigmaI, size=n_obs)

            return dict(observations=observations)

        return observation_model

    def define_adapter(self) -> Type:
        """
        Define adapter object.

        Returns:
            Type: Adapter object.
        """

        adapter = (
            bf.Adapter()
            .as_set(["observations"])
            .standardize()
            .concatenate(self._model.get_param_names(), into="inference_variables")
            .concatenate(["observations"], into="summary_variables")
        )

        return adapter
