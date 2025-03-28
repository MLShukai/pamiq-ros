from collections.abc import Generator

import pytest
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

from pamiq_ros.env import ROS2Environment
from tests.helpers import TestPublisher, TestSubscriber


@pytest.fixture
def executor() -> SingleThreadedExecutor:
    """Executor fixture."""
    return SingleThreadedExecutor()


@pytest.fixture
def obs_publisher(executor: SingleThreadedExecutor) -> TestPublisher:
    """Observation publisher fixture."""
    pub = TestPublisher("/test_obs_topic", 10)
    executor.add_node(pub)
    return pub


@pytest.fixture
def action_subscriber(executor: SingleThreadedExecutor) -> TestSubscriber:
    """Action subscriber fixture."""
    sub = TestSubscriber("/test_action_topic")
    executor.add_node(sub)
    return sub


@pytest.fixture
def initial_obs() -> String:
    """Initial observation fixture."""
    msg = String()
    msg.data = "initial observation"
    return msg


class TestROS2Environment:
    """Tests for ROS2Environment."""

    @pytest.fixture
    def env(
        self, initial_obs: String
    ) -> Generator[ROS2Environment[String, String], None, None]:
        """ROS2Environment fixture."""
        env = ROS2Environment(
            node_name="test_env_node",
            obs_topic_name="/test_obs_topic",
            obs_msg_type=String,
            action_topic_name="/test_action_topic",
            action_msg_type=String,
            initial_obs=initial_obs,
            qos=10,
        )
        yield env

    @pytest.fixture
    def setup_env(self, env: ROS2Environment) -> Generator[ROS2Environment, None, None]:
        """Setup and teardown for environment."""
        env.setup()
        try:
            yield env
        finally:
            env.teardown()

    def test_properties(self, env: ROS2Environment):
        """Test property accessors."""
        assert env.obs_topic_name == "/test_obs_topic"
        assert env.action_topic_name == "/test_action_topic"
        assert env.obs_msg_type == String
        assert env.action_msg_type == String
        assert isinstance(env.node, Node)

    def test_initial_observation(self, setup_env: ROS2Environment, initial_obs: String):
        """Test initial observation value is correctly set and returned."""
        observation = setup_env.observe()
        assert observation.data == initial_obs.data

    def test_observe_updates(
        self,
        setup_env: ROS2Environment[String, String],
        executor: SingleThreadedExecutor,
        obs_publisher: TestPublisher,
    ):
        """Test that observations are updated when new messages arrive."""
        # Publish a new message
        obs_publisher.publish("updated observation")
        executor.spin_once(0.1)

        # Get observation, should be updated
        observation = setup_env.observe()
        assert observation.data == "updated observation"

    def test_affect(
        self,
        setup_env: ROS2Environment,
        executor: SingleThreadedExecutor,
        action_subscriber: TestSubscriber,
    ):
        """Test publishing action."""
        assert action_subscriber.last_msg is None
        assert action_subscriber.received_count == 0

        msg = String()
        msg.data = "test action"
        setup_env.affect(msg)
        executor.spin_once(0.1)
        assert action_subscriber.last_msg == "test action"
        assert action_subscriber.received_count == 1

    def test_observe_before_setup(self, env: ROS2Environment):
        """Test calling observe before setup raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Environment not set up"):
            env.observe()

    def test_multiple_observations(
        self,
        setup_env: ROS2Environment[String, String],
        executor: SingleThreadedExecutor,
        obs_publisher: TestPublisher,
    ):
        """Test multiple sequential observations and updates."""
        # Initial observation
        observation1 = setup_env.observe()
        assert observation1.data == "initial observation"

        # First update
        obs_publisher.publish("observation 1")
        executor.spin_once(0.1)
        observation2 = setup_env.observe()
        assert observation2.data == "observation 1"

        # Second update
        obs_publisher.publish("observation 2")
        executor.spin_once(0.1)
        observation3 = setup_env.observe()
        assert observation3.data == "observation 2"
