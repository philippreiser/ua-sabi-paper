from typing import Any, Type


class ABIBuilder:
    """
    An ABI builder class to build the correct ABI for the specific model.
    """

    def __init__(self, config: Any) -> None:
        """
        Initialize the ABIBuilder.

        Args:
            config (Any): Dictionary containing configuration settings, including paths and dataset details.
        """

        self._config = config

    def get_abi(self, model: Type, surrogate: Type) -> Type:
        """
        Load correct ABI.

        Args:
            model (Type): The simulation model for which ABI should be performed.
            surrogate (Type): The surrogate model to be used for ABI.

        Returns:
            Type: The correct ABI.
        """

        if self._config.model.lower() == "logistic":
            from .logistic import ABI_Logistic
            return ABI_Logistic(self._config, model, surrogate)
        elif self._config.model.lower() == "log_sin":
            from .log_sin import ABI_LogSin
            return ABI_LogSin(self._config, model, surrogate)
        elif self._config.model.lower() == "plane2d":
            from .plane2d import ABI_Plane2D
            return ABI_Plane2D(self._config, model, surrogate)
        elif self._config.model.lower() == "ishigami":
            from .ishigami import ABI_Ishigami
            return ABI_Ishigami(self._config, model, surrogate)
        elif self._config.model.lower() == "co2":
            from .co2 import ABI_CO2
            return ABI_CO2(self._config, model, surrogate)
        elif self._config.model.lower() == "micp":
            from .micp import ABI_MICP
            return ABI_MICP(self._config, model, surrogate)
        else:
            raise ValueError("Model for ABI not supported.")
