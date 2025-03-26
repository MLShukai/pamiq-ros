import threading
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, override

import rclpy
from pamiq_core.interaction import Environment
from rclpy.qos import QoSProfile

type Kwds = Mapping[str, Any]
DEFAULT_KWDS: Kwds = MappingProxyType({})


class ROS2Environment[Obs, Act](Environment[Obs, Act]):
    """Environment implementation for ROS2 integration.

    This class provides an Environment interface for ROS2, allowing
    pamiq-core to interact with ROS2 nodes. It subscribes to observation
    topics and publishes action topics.
    """

    def __init__(
        self,
        node_name: str,
        obs_topic_name: str,
        obs_msg_type: type[Obs],
        action_topic_name: str,
        action_msg_type: type[Act],
        qos: int | QoSProfile,
        node_kwds: Kwds = DEFAULT_KWDS,
        obs_kwds: Kwds = DEFAULT_KWDS,
        action_kwds: Kwds = DEFAULT_KWDS,
        obs_get_timeout: float | None = None,
    ) -> None:
        """Initialize ROS2 environment.

        Args:
            node_name: Name of the ROS2 node
            obs_topic_name: Topic name for observations
            obs_msg_type: Message type for observations
            action_topic_name: Topic name for actions
            action_msg_type: Message type for actions
            qos: Quality of Service profile or depth
            node_kwds: Additional keyword arguments for node creation
            obs_kwds: Additional keyword arguments for subscription
            action_kwds: Additional keyword arguments for publisher
            obs_get_timeout: Timeout for waiting for observations (None means wait forever)
        """
        super().__init__()
        self._node = rclpy.create_node(node_name, **node_kwds)
        self._obs_subscriber = self._node.create_subscription(  # pyright: ignore[reportUnknownMemberType]
            obs_msg_type,
            obs_topic_name,
            self._obs_callback,
            qos,
            **obs_kwds,
        )
        self._obs_get_event = threading.Event()
        self._obs_get_timeout = obs_get_timeout
        self._obs_lock = threading.RLock()
        self._obs_thread: threading.Thread | None = None
        self._observation: Obs
        # _observation is intentionally not initialized here
        # It will be set in the callback when the first observation arrives

        self._action_pubisher = self._node.create_publisher(  # pyright: ignore[reportUnknownMemberType]
            action_msg_type, action_topic_name, qos, **action_kwds
        )

    def _obs_callback(self, data: Obs) -> None:
        """Callback function for observation subscription.

        Args:
            data: Observation data from ROS2
        """
        with self._obs_lock:
            self._observation = data
            if not self._obs_get_event.is_set():
                self._obs_get_event.set()

    @override
    def observe(self) -> Obs:
        """Get observation from ROS2.

        This method waits for the first observation to arrive if none has been
        received yet. After the first observation, it returns the most recent
        observation immediately.

        Returns:
            Current observation from ROS2

        Raises:
            TimeoutError: If observation not received within timeout
        """
        if self._obs_thread is None:
            raise RuntimeError("Please call `setup` before.")

        if self._obs_get_event.wait(self._obs_get_timeout):
            with self._obs_lock:
                return self._observation
        else:
            raise TimeoutError(
                f"Timed out waiting for observation on topic '{self._obs_subscriber.topic_name}'"  # pyright: ignore[reportUnknownMemberType]
            )

    @override
    def affect(self, action: Act) -> None:
        """Publish action to ROS2.

        Args:
            action: Action message to publish
        """
        self._action_pubisher.publish(action)

    @override
    def setup(self) -> None:
        """Set up ROS2 node and start spinning it in a separate thread."""
        super().setup()
        self._obs_thread = threading.Thread(
            target=lambda: rclpy.spin(self._node), daemon=True
        )
        self._obs_thread.start()

    @override
    def teardown(self) -> None:
        """Clean up ROS2 resources."""
        super().teardown()
        self._node.destroy_node()
        self._obs_thread = None
