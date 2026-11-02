"""Index build events and a worker that keeps the UI responsive during long stages."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from queue import Empty, Queue


@contextmanager
def index_build_lock(index):
    """Exclude concurrent builds of the same project across threads and processes."""
    from .index import IndexErrorBase

    index.validate_project()
    index.db_path.parent.mkdir(parents=True, exist_ok=True)
    with (index.db_path.parent / "index-build.lock").open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"\0")
            lock.flush()
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise IndexErrorBase("该项目正在构建索引，请等待当前构建完成后重试。") from exc
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def emit_index_progress(
    callback, stage, message, *, completed=None, total=None, unit="", current_file=None
):
    if callback is None:
        return
    event = {"stage": stage, "message": message, "completed": completed, "total": total, "unit": unit}
    if current_file is not None:
        event["current_file"] = current_file
    callback(event)


def run_index_build(build, progress, *, interval=0.5):
    """Run build(callback) in a worker; deliver events and heartbeats on the calling thread."""
    events = Queue()
    started = phase_started = time.monotonic()
    current = {"stage": "start", "message": "正在准备索引构建…", "completed": None, "total": None}

    def notify():
        now = time.monotonic()
        progress(
            {
                **current,
                "elapsed_seconds": round(now - started, 1),
                "phase_seconds": round(now - phase_started, 1),
            }
        )

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="index-build") as executor:
        future = executor.submit(build, events.put)
        notify()
        while True:
            try:
                event = events.get(timeout=interval)
            except Empty:
                if future.done():
                    break
                notify()
                continue
            if event["stage"] != current["stage"]:
                phase_started = time.monotonic()
            current = event
            notify()
        return future.result()
