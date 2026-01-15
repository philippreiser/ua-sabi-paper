from typing import Any, Type


class SurrogateBuilder:
    """
    A surrogate builder class to build the correct surrogate to perform SABI.
    """

    def __init__(self, config: Any, model: Type, logger: Type = None) -> None:
        """
        Initialize the Surrogatebuilder.

        Args:
            config: Dictionary containing configuration settings, including paths and dataset details.
            model: True model to build surrogate for.
            logger: logger.
        """

        self._config = config
        self._model = model
        self._logger = logger

    def build_surrogate(self) -> Type:
        """
        Build surrogate model as specified in config.
        """

        if self._config.surrogate == "bayesian_apc":
            from .bayesian_apc import BayesianaPC
            return BayesianaPC(
                self._config.bayesian_apc.polynomial_degree,
                self._config.bayesian_apc.log3_n_training_points,
                self._config.bayesian_apc.n_samples,
                self._config.bayesian_apc.n_warmup,
                self._config.bayesian_apc.n_chains,
                self._config.bayesian_apc.prop_sigma_approx,
                self._config.bayesian_apc.use_sparse_pce,
                self._config.output_path,
                self._model,
                self._logger)
        elif self._config.surrogate == "bayesian_point_apc":
            from .bayesian_apc import BayesianPointaPC
            return BayesianPointaPC(
                self._config.bayesian_apc.polynomial_degree,
                self._config.bayesian_apc.log3_n_training_points,
                self._config.bayesian_apc.n_samples,
                self._config.bayesian_apc.n_warmup,
                self._config.bayesian_apc.n_chains,
                self._config.bayesian_apc.prop_sigma_approx,
                self._config.bayesian_apc.use_sparse_pce,
                self._config.output_path,
                self._model,
                self._logger)
        elif self._config.surrogate == "bayesian_pce":
            from .bayesian_pce import BayesianPCE
            return BayesianPCE(
                self._config.bayesian_pce.polynomial_degree,
                self._config.bayesian_pce.log2_n_training_points,
                self._config.bayesian_pce.n_samples,
                self._config.bayesian_pce.n_warmup,
                self._config.bayesian_pce.n_chains,
                self._config.bayesian_pce.prop_sigma_approx,
                self._config.output_path,
                self._model)
        elif self._config.surrogate == "bayesian_point_pce":
            from .bayesian_pce import BayesianPointPCE
            return BayesianPointPCE(
                self._config.bayesian_pce.polynomial_degree,
                self._config.bayesian_pce.log2_n_training_points,
                self._config.bayesian_pce.n_samples,
                self._config.bayesian_pce.n_warmup,
                self._config.bayesian_pce.n_chains,
                self._config.bayesian_pce.prop_sigma_approx,
                self._config.output_path,
                self._model)
        elif self._config.surrogate == "point_gp":
            from .gp import PointGP
            return PointGP(
                self._config.gp.log2_n_training_points,
                self._config.gp.variance,
                self._config.gp.lengthscale,
                self._config.gp.measurement_noise,
                self._config.gp.prop_sigma_approx,
                self._model)
        else:
            raise ValueError("Surrogatebuilder not supported.")
