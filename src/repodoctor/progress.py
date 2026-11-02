"""Incremental public activity events; never expose model reasoning fields."""

from __future__ import annotations

import asyncio
import time
from contextlib import suppress


class ProgressReporter:
    def __init__(self, callback=None, *, started=None, interval=3):
        self.callback = callback
        self.started = time.monotonic() if started is None else started
        self.interval = interval
        self.events = []
        self.current = None
        self.phase_started = self.started
        self.task = None

    def emit(self, event):
        now = time.monotonic()
        item = {**event, "elapsed_seconds": round(now - self.started, 1)}
        self.current = item
        self.phase_started = now
        self.events.append(item)
        if self.callback:
            self.callback(item)

    async def _heartbeat(self):
        while True:
            await asyncio.sleep(self.interval)
            if self.current and self.callback:
                self.callback(
                    {
                        "event": "waiting",
                        "message": self.current["message"],
                        "elapsed_seconds": round(time.monotonic() - self.started, 1),
                        "phase_seconds": round(time.monotonic() - self.phase_started, 1),
                    }
                )

    async def __aenter__(self):
        if self.callback:
            self.task = asyncio.create_task(self._heartbeat())
        return self

    async def __aexit__(self, *args):
        if self.task:
            self.task.cancel()
            with suppress(asyncio.CancelledError):
                await self.task
