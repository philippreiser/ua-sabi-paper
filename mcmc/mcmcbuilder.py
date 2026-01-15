from typing import Any, Type


class MCMCBuilder:
    """
    MCMC inference builder class.
    """

    def __init__(self, config: Any, model: Type, surrogate: Type) -> None:
        """
        Initialize the MCMC builder.

        Args:
            config: Dictionary containing configuration settings, including paths and dataset details.
            model (Type): True model.
            surrogate: Trained surrogate to perform inference with.
        """

        self._config = config
        self._surrogate = surrogate
        self._model = model

    def build_mcmc(self) -> Type:
        """
        Build MCMC inference for surrogate as specified in config.
        """

        if self._config.surrogate == "true_model":
            if self._config.model.lower() == "logistic":
                from .logistic_mcmc import LogisticMCMC
                return LogisticMCMC(
                    int(self._config.out.n_param_samples /
                        self._config.mcmc.n_chains),
                    self._config.mcmc.n_chains,
                    self._config.mcmc.n_warmup,
                    self._model)
            elif self._config.model.lower() == "log_sin":
                from .log_sin_mcmc import LogSinMCMC
                return LogSinMCMC(
                    int(self._config.out.n_param_samples /
                        self._config.mcmc.n_chains),
                    self._config.mcmc.n_chains,
                    self._config.mcmc.n_warmup,
                    self._model)
            elif self._config.model.lower() == "plane2d":
                from .plane2d_mcmc import Plane2DMCMC
                return Plane2DMCMC(
                    int(self._config.out.n_param_samples /
                        self._config.mcmc.n_chains),
                    self._config.mcmc.n_chains,
                    self._config.mcmc.n_warmup,
                    self._model)
            else:
                raise ValueError(
                    f"MCMCBuilder not supported for {self._config.model}.")
        elif self._config.surrogate == "bayesian_pce":
            from .bayesian_pce_mcmc import BayesianPCEMCMC
            return BayesianPCEMCMC(
                int(self._config.out.n_param_samples /
                    self._config.mcmc.n_chains),
                self._config.mcmc.n_chains,
                self._config.mcmc.n_warmup,
                self._surrogate)
        elif self._config.surrogate == "bayesian_point_pce":
            from .bayesian_pce_mcmc import BayesianPointPCEMCMC
            return BayesianPointPCEMCMC(
                int(self._config.out.n_param_samples /
                    self._config.mcmc.n_chains),
                self._config.mcmc.n_chains,
                self._config.mcmc.n_warmup,
                self._surrogate)
        elif self._config.surrogate == "bayesian_apc":
            from .bayesian_apc_mcmc import BayesianaPCMCMC
            return BayesianaPCMCMC(
                int(self._config.out.n_param_samples /
                    self._config.mcmc.n_chains),
                self._config.mcmc.n_chains,
                self._config.mcmc.n_warmup,
                self._surrogate)
        elif self._config.surrogate == "bayesian_point_apc":
            from .bayesian_apc_mcmc import BayesianPointaPCMCMC
            return BayesianPointaPCMCMC(
                int(self._config.out.n_param_samples /
                    self._config.mcmc.n_chains),
                self._config.mcmc.n_chains,
                self._config.mcmc.n_warmup,
                self._surrogate)
        else:
            raise ValueError("MCMCBuilder not supported.")
