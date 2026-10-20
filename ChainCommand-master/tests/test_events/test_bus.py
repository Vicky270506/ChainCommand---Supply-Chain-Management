"""Tests for EventBus queued processing."""

from __future__ import annotations

import asyncio

import pytest

from chaincommand.data.schemas import AlertSeverity, SupplyChainEvent
from chaincommand.events.bus import EventBus


class TestEventBusLifecycle:
    @pytest.mark.asyncio
    async def test_enqueue_processes_after_start(self):
        bus = EventBus()
        received: list[SupplyChainEvent] = []
        done = asyncio.Event()

        async def handler(event: SupplyChainEvent) -> None:
            received.append(event)
            done.set()

        bus.subscribe("tick", handler)
        await bus.start()
        await bus.enqueue(
            SupplyChainEvent(
                event_type="tick",
                severity=AlertSeverity.LOW,
                source_agent="test",
                description="queued event",
            )
        )

        await asyncio.wait_for(done.wait(), timeout=5.0)

        await bus.stop()
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self):
        bus = EventBus()
        await bus.start()
        first_task = bus._task
        await bus.start()
        await bus.stop()

        assert first_task is not None
        assert bus._task is None
