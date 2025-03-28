import threading
import time
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
    def env_with_timeout(
        self, initial_obs: String
    ) -> Generator[ROS2Environment[String, String], None, None]:
        """ROS2Environment fixture with observation timeout."""
        env = ROS2Environment(
            node_name="test_env_timeout_node",
            obs_topic_name="/test_obs_topic",
            obs_msg_type=String,
            action_topic_name="/test_action_topic",
            action_msg_type=String,
            initial_obs=initial_obs,
            qos=10,
            obs_timeout=0.2,  # Short timeout for testing
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

    @pytest.fixture
    def setup_env_with_timeout(
        self, env_with_timeout: ROS2Environment
    ) -> Generator[ROS2Environment, None, None]:
        """Setup and teardown for environment with timeout."""
        env_with_timeout.setup()
        try:
            yield env_with_timeout
        finally:
            env_with_timeout.teardown()

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

    def test_observe_with_timeout(
        self,
        setup_env_with_timeout: ROS2Environment[String, String],
    ):
        """Test observation with timeout returns the current observation when
        no new data arrives."""
        # First observation returns initial value
        start_time = time.time()
        observation = setup_env_with_timeout.observe()
        elapsed = time.time() - start_time

        # Should return quickly with the initial value
        assert observation.data == "initial observation"
        # Should have waited for the timeout
        assert 0.2 < elapsed < 0.3  # Loose upper bound

    def test_waiting_for_observation(
        self,
        setup_env_with_timeout: ROS2Environment[String, String],
        executor: SingleThreadedExecutor,
        obs_publisher: TestPublisher,
    ):
        """Test that observe waits for new observations within timeout
        period."""

        # Start a background thread that will publish after a short delay
        def delayed_publish():
            obs_publisher.publish("delayed observation")
            executor.spin_once(0.05)

        # Start background thread and call observe
        bg_thread = threading.Timer(0.1, delayed_publish)
        bg_thread.start()

        # This should wait for the new observation
        start = time.perf_counter()
        observation = setup_env_with_timeout.observe()
        assert 0.1 < time.perf_counter() - start < 0.2
        # Wait for background thread to complete
        bg_thread.join()

        # Should receive the published message
        assert observation.data == "delayed observation"

    def test_shutdown_notification(
        self,
        env_with_timeout: ROS2Environment[String, String],
    ):
        """Test that observe is notified when ROS2 node is shut down."""
        env_with_timeout.setup()

        # Start a background thread that will call observe
        result: dict[str, String | None] = {"observation": None}

        def observe_thread():
            result["observation"] = env_with_timeout.observe()

        bg_thread = threading.Thread(target=observe_thread)
        bg_thread.start()

        # Give the thread a moment to start and begin waiting
        time.sleep(0.1)

        # Shutdown the environment
        env_with_timeout.teardown()

        # Wait for the background thread to complete
        bg_thread.join(timeout=1.0)
        assert not bg_thread.is_alive()  # Thread should have completed

        # Observation should be the initial one (or whatever was last)
        assert result["observation"] is not None
        assert result["observation"].data == "initial observation"
