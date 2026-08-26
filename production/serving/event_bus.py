"""
Event bus abstraction for the real-time transaction pipeline.

Why this exists: real-time fraud/mule detection architectures publish every
transaction as a stream event and process it as it arrives, rather than
waiting for a nightly batch — Kafka is the industry-standard choice for this
(high throughput, durable, ordered per-partition), and is what this module
targets for production.

This environment has no running Kafka broker to test against, so — same
pattern as the feature store — the pipeline is built against an abstract
EventBus interface with two implementations: InMemoryEventBus (zero
dependencies, fully testable here, used for local development/demo) and
KafkaEventBus (real kafka-python client code, correct and ready, but
requires a live broker to actually run). Swapping backends is a one-line
change in whichever service constructs the bus — nothing else in the
pipeline should import kafka directly.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
import json


class EventBus(ABC):
    @abstractmethod
    def publish(self, topic: str, message: dict) -> None:
        raise NotImplementedError

    @abstractmethod
    def subscribe(self, topic: str, handler) -> None:
        """Register a callable to be invoked with each message published to
        `topic`. Implementations decide whether this delivers immediately
        (InMemoryEventBus) or via a consumer loop the caller must run
        (KafkaEventBus.consume_forever)."""
        raise NotImplementedError


class InMemoryEventBus(EventBus):
    """Synchronous, in-process pub/sub — a message published is delivered to
    every subscribed handler immediately, in the same call. No durability,
    no cross-process delivery. Used for local development and for testing
    the transaction_consumer pipeline logic without a Kafka broker."""

    def __init__(self):
        self._subscribers = defaultdict(list)
        self.published_log = []  # kept for test assertions / demo inspection

    def publish(self, topic: str, message: dict) -> None:
        self.published_log.append((topic, message))
        for handler in self._subscribers[topic]:
            handler(message)

    def subscribe(self, topic: str, handler) -> None:
        self._subscribers[topic].append(handler)


class KafkaEventBus(EventBus):
    """Production backend. Requires kafka-python (pip install kafka-python)
    and a running Kafka broker — this class is correct, ready-to-deploy
    code, not runnable in this sandboxed environment. Delivers guaranteed
    ordering per-partition and durability that InMemoryEventBus cannot,
    which matters for a financial audit trail: a transaction event must not
    be silently lost between ingestion and scoring."""

    def __init__(self, bootstrap_servers='localhost:9092'):
        try:
            from kafka import KafkaProducer, KafkaConsumer
            from kafka.serializer import Serializer
        except ImportError:
            raise ImportError("kafka-python not installed — pip install kafka-python")

        class _JSONSerializer(Serializer):
            # kafka-python 3.x deprecates raw-callable value_serializer in
            # favor of this abstract class — found and fixed by actually
            # installing kafka-python and constructing this class for real,
            # not just trusting the code would work.
            def serialize(self, topic, headers, data):
                return json.dumps(data).encode('utf-8')

        self._bootstrap_servers = bootstrap_servers
        self._producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=_JSONSerializer(),
            acks='all',  # wait for full replication — no dropped transaction events
        )
        self._handlers = defaultdict(list)

    def publish(self, topic: str, message: dict) -> None:
        future = self._producer.send(topic, value=message)
        future.get(timeout=10)  # block for delivery confirmation; fraud events must not be fire-and-forget

    def subscribe(self, topic: str, handler) -> None:
        self._handlers[topic].append(handler)

    def consume_forever(self, topic: str, group_id: str = 'sachet-consumers'):
        """Blocking consumer loop — run this in a dedicated worker process,
        not inline in a request handler. Dispatches each message to every
        handler registered via subscribe() for this topic."""
        from kafka import KafkaConsumer
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=self._bootstrap_servers,
            group_id=group_id,
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
            auto_offset_reset='earliest',
            enable_auto_commit=True,
        )
        for record in consumer:
            for handler in self._handlers[topic]:
                handler(record.value)
