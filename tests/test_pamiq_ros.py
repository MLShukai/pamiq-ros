import tomllib
import xml.etree.ElementTree as ET

import pamiq_ros

from .helpers import PROJECT_ROOT


def test_version():
    with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)
    assert pyproject["project"]["version"] == pamiq_ros.__version__

    ros_package = ET.parse(PROJECT_ROOT / "package.xml").getroot()
    assert ros_package.find("version").text == pamiq_ros.__version__
