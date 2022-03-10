from typing import Dict, Any


class Singleton(type):  # type: ignore
    """ A metaclass that creates a Singleton base class when called. """

    _instances: Dict[Any, Any] = {}

    def __call__(cls, *args, **kwargs) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(
                *args, **kwargs
            )
        return cls._instances[cls]
