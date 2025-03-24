import threading
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, override

import rclpy
from pamiq_core.interaction import Environment
from rclpy.qos import QoSProfile

type Kwds = Mapping[str, Any]
DEFAULT_KWDS: Kwds = MappingProxyType({})


class ROS2Env[Obs, Act](Environment[Obs, Act]):
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

        self._action_pubisher = self._node.create_publisher(  # pyright: ignore[reportUnknownMemberType]
            action_msg_type, action_topic_name, qos, **action_kwds
        )

    def _obs_callback(self, data: Obs) -> None:
        with self._obs_lock:
            self._observation = data
            if not self._obs_get_event.is_set():
                self._obs_get_event.set()

    @override
    def observe(self) -> Obs:
        if self._obs_get_event.wait(self._obs_get_timeout):
            with self._obs_lock:
                return self._observation
        else:
            raise TimeoutError

    @override
    def affect(self, action: Act) -> None:
        self._action_pubisher.publish(action)

    @override
    def setup(self) -> None:
        super().setup()
        self._obs_thread = threading.Thread(target=rclpy.spin, args=(self._node,))
        self._obs_thread.start()

    @override
    def teardown(self) -> None:
        super().teardown()
        self._node.destroy_node()
        if self._obs_thread is not None:
            self._obs_thread.join()
