from importlib import metadata

from .env import ROS2Environment

__version__ = metadata.version("pamiq-ros")


__all__ = [
    "ROS2Environment",
]
