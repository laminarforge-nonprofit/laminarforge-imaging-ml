"""MQTT publisher for per-chip imaging metrics.

Publishes JSON to topic ``imaging/metrics``. Broker configured from env vars:
    LF_MQTT_BROKER  (default 100.119.87.128:1883)
    LF_MQTT_USER    (optional)
    LF_MQTT_PASS    (optional)
    LF_MQTT_TOPIC   (default imaging/metrics)

The broker ACL in A-BCBDCDC3 needs a user ``lf-imaging`` with publish rights to
``imaging/#``. Document this in the README for the MQTT admin.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_BROKER = "100.119.87.128:1883"
DEFAULT_TOPIC = "imaging/metrics"


def _parse_broker(spec: str) -> tuple[str, int]:
    host, _, port = spec.partition(":")
    return host, int(port) if port else 1883


def _default_serializer(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if is_dataclass(obj):
        return asdict(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


class MetricsPublisher:
    """Small wrapper around paho-mqtt for fire-and-forget metric publishing."""

    def __init__(
        self,
        broker: str | None = None,
        topic: str = DEFAULT_TOPIC,
        username: str | None = None,
        password: str | None = None,
        client_id: str = "lf-imaging",
    ) -> None:
        import paho.mqtt.client as mqtt

        self.topic = os.getenv("LF_MQTT_TOPIC", topic)
        host, port = _parse_broker(broker or os.getenv("LF_MQTT_BROKER", DEFAULT_BROKER))
        self._client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id,
        )
        user = username or os.getenv("LF_MQTT_USER")
        pw = password or os.getenv("LF_MQTT_PASS")
        if user:
            self._client.username_pw_set(user, pw)
        self._host = host
        self._port = port
        self._connected = False

    def connect(self, timeout: int = 5) -> None:
        self._client.connect(self._host, self._port, keepalive=timeout * 3)
        self._client.loop_start()
        self._connected = True

    def publish(self, payload: Any) -> None:
        """Publish any dataclass / dict / JSON-safe object."""
        if not self._connected:
            self.connect()
        body = json.dumps(
            asdict(payload) if is_dataclass(payload) else payload,
            default=_default_serializer,
        )
        result = self._client.publish(self.topic, body, qos=1)
        # Avoid hard-blocking the pipeline if the broker is slow; log instead.
        if result.rc != 0:
            log.warning("MQTT publish rc=%s topic=%s", result.rc, self.topic)

    def close(self) -> None:
        if self._connected:
            self._client.loop_stop()
            self._client.disconnect()
            self._connected = False

    def __enter__(self) -> MetricsPublisher:
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def publish_metrics(metrics: Any, **kwargs: Any) -> None:
    """Convenience one-shot publish — opens a connection, publishes, closes."""
    with MetricsPublisher(**kwargs) as pub:
        pub.publish(metrics)
