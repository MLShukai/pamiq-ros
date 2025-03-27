from collections.abc import Generator

import pytest
import rclpy


@pytest.fixture(autouse=True)  # Using this fixture in all test session.
def rclpy_init() -> Generator[None, None, None]:
    rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()
