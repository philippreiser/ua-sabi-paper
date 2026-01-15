from typing import Any, Dict, Type
import os
import numpy as np
import keras
import bayesflow as bf

from .abi_base import ABI
from surrogate_methods.bayesian_pce import BayesianPCEBase
from surrogate_methods.bayesian_apc import BayesianaPCBase
from simulation_models.micp import MICP


class ABI_MICP(ABI):
    """
    Perform Amortized Bayesian inference.
    """

    def __init__(self, config: Any, model: MICP, surrogate: Type) -> None:
        """
        Initialize ABI for MICP. (Works also with bayesflow 2.0.5)

        Args:
            config (Dict): Dictionary containing configuration settings, including paths and datasets details.
            model (Type): True model.
            surrogate (Type): Surrogate model.
        """

        super().__init__(config, model, surrogate)

    def define_adapter(self) -> Type:
        """
        Define adapter object.

        Returns:
            Type: Adapter object.
        """

        adapter = (
            bf.Adapter()
            # .standardize() # for old bf version
        )

        # constrain to parameter boundaries
        bounds = self._model.get_param_bounds()
        for param, (lower, upper) in bounds.items():
            adapter = adapter.constrain(param, lower=lower, upper=upper)

        if self._config.abi.sumnet == "none":
            adapter = (
                adapter
                .concatenate(self._model.get_param_names(), into="inference_variables")
                .concatenate(["observations"], into="inference_conditions")
            )
        else:
            adapter = (
                adapter
                .as_set(["observations"]) # for deep set summary network
                .concatenate(self._model.get_param_names(), into="inference_variables")
                .concatenate(["observations"], into="summary_variables")
            )

        return adapter
