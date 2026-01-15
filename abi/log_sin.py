from typing import Any, Dict, Type
import numpy as np
import keras
import bayesflow as bf

from .abi_base import ABI


class ABI_LogSin(ABI):
    """
    Perform Amortized Bayesian inference.
    """

    def __init__(self, config: Any, model: Type, surrogate: Type) -> None:
        """
        Initialize ABI for Logistic.

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

        def observation_model(w: np.ndarray, n_obs: int) -> Dict:
            """
            Evaluate the LogSin observation model at a given parameter value.

            Args:
                w(np.ndarray): Parameter value.
                n_obs(int): Number of observations.

            Returns:
                Dict: Dictionary containing the (inputs and) model output.
            """

            inputs = self._model.inputs(n_obs)
            out = self._surrogate.evaluate(np.repeat(w, n_obs), inputs)
            observations = np.random.normal(
                out, self._model._sigmaI, size=n_obs)

            return dict(inputs=inputs, observations=observations)

        return observation_model

    def define_adapter(self) -> Type:
        """
        Define adapter object.

        Returns:
            Type: Adapter object.
        """

        adapter = (
            bf.Adapter()
            .broadcast("n_obs", to="inputs")
            .as_set(["inputs", "observations"])
            .standardize(exclude=["n_obs"])
            .sqrt("n_obs")
            .concatenate(["w"], into="inference_variables")
            .concatenate(["inputs", "observations"], into="summary_variables")
            .rename("n_obs", "inference_conditions")
        )

        return adapter
