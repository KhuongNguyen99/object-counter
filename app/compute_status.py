"""Read-only RunPod compute status and persisted terminal job activity."""

import json
import logging
import os
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .runpod_client import RunPodClient


logger = logging.getLogger(__name__)
IDLE_TIMEOUT_SECONDS = 600
HEALTH_CACHE_SECONDS = 10


def _utc_now():
    return datetime.now(timezone.utc)


def _iso(value):
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def _count(group, key):
    value = group.get(key)
    if type(value) is not int or value < 0:
        raise ValueError("Invalid RunPod health count")
    return value


class ComputeStatus:
    def __init__(self, data_dir, endpoint_id, *, client_factory=None, now=None, monotonic=None):
        self.endpoint_id = endpoint_id
        self.path = Path(data_dir) / "runpod-activity.json"
        self._client_factory = client_factory or RunPodClient
        self._client = None
        self._now = now or _utc_now
        self._monotonic = monotonic or time.monotonic
        self._activity_lock = threading.Lock()
        self._health_lock = threading.Lock()
        self._active = 0
        self._last_finished = self._load_finished()
        self._restored_finished = self._last_finished is not None
        self._health = None
        self._health_expires = 0.0

    def _load_finished(self):
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("endpoint_id") != self.endpoint_id:
                return None
            value = datetime.fromisoformat(data["completed_at"].replace("Z", "+00:00"))
            return value.astimezone(timezone.utc) if value.tzinfo else None
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            return None

    def _persist_finished(self):
        temporary = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.path.parent,
                                             prefix=".runpod-activity-", delete=False) as file:
                temporary = Path(file.name)
                json.dump({"endpoint_id": self.endpoint_id,
                           "completed_at": (_iso(self._last_finished)
                                            if self._last_finished is not None else None)}, file)
            os.replace(temporary, self.path)
        except OSError:
            logger.warning("Could not persist RunPod activity")
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    def start(self):
        with self._activity_lock:
            self._active += 1
            self._last_finished = None
            self._restored_finished = False
            self._persist_finished()

    def finish(self, terminal=False):
        with self._activity_lock:
            self._active = max(0, self._active - 1)
            if terminal:
                self._last_finished = self._now()
                self._restored_finished = False
                self._persist_finished()

    def _read_health(self):
        with self._health_lock:
            now = self._monotonic()
            if self._health is not None and now < self._health_expires:
                return self._health
            try:
                if self._client is None:
                    self._client = self._client_factory()
                raw = self._client.health()
                workers, jobs = raw["workers"], raw["jobs"]
                if not isinstance(workers, dict) or not isinstance(jobs, dict):
                    raise ValueError("Invalid RunPod health response")
                self._health = (_count(workers, "running"), _count(workers, "initializing"),
                                _count(jobs, "inProgress"), _count(jobs, "inQueue"))
            except Exception:
                self._health = ("unavailable",)
            self._health_expires = now + HEALTH_CACHE_SECONDS
            return self._health

    def snapshot(self):
        health = self._read_health()
        now = self._now()
        with self._activity_lock:
            if (health[0] != "unavailable" and (health[2] or health[3]) and
                    self._restored_finished):
                self._last_finished = None
                self._restored_finished = False
                self._persist_finished()
            active, last_finished = self._active, self._last_finished
        deadline = None
        if health[0] == "unavailable":
            state = "starting" if active else "unavailable"
        else:
            running, initializing, in_progress, in_queue = health
            if in_progress or (active and running):
                state = "busy"
            elif in_queue or active or initializing:
                state = "starting"
            elif running:
                state = "ready"
                if last_finished is not None:
                    deadline = _iso(last_finished + timedelta(seconds=IDLE_TIMEOUT_SECONDS))
            else:
                state = "sleeping"
        return {"enabled": True, "state": state, "server_time": _iso(now),
                "idle_deadline": deadline, "idle_timeout_seconds": IDLE_TIMEOUT_SECONDS}

    def close(self):
        with self._health_lock:
            if self._client is not None:
                self._client.close()

