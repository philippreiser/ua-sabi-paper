from typing import Any, Type


class ModelLoader:
    """
    A model loader class to load the correct model for which SABI should be performed.
    """

    def __init__(self, config: Any) -> None:
        """
        Initialize the Model loader.

        Args:
            config: Dictionary containing configuration settings, including paths and dataset details.
        """

        self._config = config

    def get_model(self) -> Type:
        """
        Load correct simulation model.

        Returns:
            Type: The correct simulation model.
        """

        if self._config.model.lower() == "logistic":
            from .logistic import Logistic
            return Logistic(self._config, self._config.logistic.w_range)
        elif self._config.model.lower() == "log_sin":
            from .log_sin import LogSin
            return LogSin(self._config, self._config.log_sin.w_range, self._config.log_sin.x_range)
        elif self._config.model.lower() == "plane2d":
            from .plane2d import Plane2D
            return Plane2D(self._config, self._config.plane2d.w_range, self._config.plane2d.x_range)
        elif self._config.model.lower() == "ishigami":
            from .ishigami import Ishigami
            return Ishigami(self._config, self._config.ishigami.a_range, self._config.ishigami.b_range)
        elif self._config.model.lower() == "co2":
            from .co2 import CO2
            return CO2(self._config)
        elif self._config.model.lower() == "sir":
            from .sir import SIR
            return SIR(self._config, self._config.sir.beta_range, self._config.sir.gamma_range, self._config.sir.t_range)
        elif self._config.model.lower() == "micp":
            from .micp import MICP
            return MICP(self._config)
        else:
            raise ValueError("Model not supported.")
