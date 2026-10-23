"""RunPod Serverless transport and the CountGD++ remote inference adapter.

The queue API accepts one image per job. Image bytes are sent unchanged; the
worker owns inference and this module validates its result and draws locally.
"""

import base64
import json
import math
import os
from pathlib import Path
import re
import time
from urllib import error, request
from uuid import uuid4

from .models import COUNTGD_MODEL_ID, FASTSAM_MODEL_ID


# RunPod's documented maximum for the asynchronous /run request body.
MAX_RUN_PAYLOAD_BYTES = 10_000_000
MAX_SYNC_PAYLOAD_BYTES = 20_000_000
MAX_RESPONSE_BYTES = 20_000_000
MAX_DETECTIONS = 30_000
_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_TRANSIENT_HTTP_CODES = frozenset((408, 429, 500, 502, 503, 504))
_MAX_STATUS_FAILURES = 4
_WARNING = "Experimental model; review overlapping objects."
_LIMIT_WARNING = " Detection limit reached in an image or tile; the count may be low."
_REFERENCE_WARNING = "No object candidates found. Draw a box manually."


class RunPodError(RuntimeError):
    """A sanitized remote transport or worker failure."""


class _TransientRunPodError(RunPodError):
    """A status read can be retried without creating another job."""


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward the bearer credential to a redirect target.
        return None


def _number(value, *, minimum=0, maximum=None):
    return (type(value) in (int, float) and math.isfinite(value) and value >= minimum
            and (maximum is None or value <= maximum))


def _box(value, width, height):
    if not isinstance(value, list) or len(value) != 4 or not all(_number(n) for n in value):
        return False
    x1, y1, x2, y2 = value
    return 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height


def _safe_metrics(value, *, api_key=""):
    """Keep known worker diagnostics, discarding arbitrary provider strings."""
    if not isinstance(value, dict):
        return {}
    numeric = ("handler_ms", "worker_ready_ms", "preload_ms", "startup_model_load_ms",
               "peak_cuda_allocated_bytes")
    clean = {key: value[key] for key in numeric if _number(value.get(key))}
    if type(value.get("cold_start")) is bool:
        clean["cold_start"] = value["cold_start"]
    if (isinstance(value.get("worker_id"), str) and _ID.fullmatch(value["worker_id"]) and
            (not api_key or api_key not in value["worker_id"])):
        clean["worker_id"] = value["worker_id"]
    if (isinstance(value.get("gpu_name"), str) and len(value["gpu_name"]) <= 100 and
            re.fullmatch(r"[A-Za-z0-9 ._+/-]+", value["gpu_name"]) and
            (not api_key or api_key not in value["gpu_name"])):
        clean["gpu_name"] = value["gpu_name"]
    if value.get("model") in (COUNTGD_MODEL_ID, FASTSAM_MODEL_ID):
        clean["model"] = value["model"]
    if (isinstance(value.get("model_sha256"), str) and
            re.fullmatch(r"[0-9a-f]{64}", value["model_sha256"])):
        clean["model_sha256"] = value["model_sha256"]
    if (isinstance(value.get("build_revision"), str) and
            (value["build_revision"] == "unknown" or
             re.fullmatch(r"[0-9a-fA-F]{7,40}", value["build_revision"]))):
        clean["build_revision"] = value["build_revision"]
    if value.get("preloaded_model") in ("count", "suggest", "none"):
        clean["preloaded_model"] = value["preloaded_model"]
    return clean


class RunPodClient:
    """RunPod client with no automatic submission or redirect retries."""

    def __init__(self, api_key=None, endpoint_id=None, timeout_seconds=None, *, opener=None):
        self.api_key = api_key if api_key is not None else os.environ.get("RUNPOD_API_KEY", "")
        self.endpoint_id = endpoint_id if endpoint_id is not None else os.environ.get("RUNPOD_ENDPOINT_ID", "")
        raw_timeout = (timeout_seconds if timeout_seconds is not None
                       else os.environ.get("RUNPOD_TIMEOUT_SECONDS", "600"))
        if not isinstance(self.api_key, str) or not self.api_key.strip():
            raise ValueError("RUNPOD_API_KEY is required for remote inference.")
        if any(ord(character) < 33 or ord(character) > 126 for character in self.api_key):
            raise ValueError("RUNPOD_API_KEY contains invalid characters.")
        if not isinstance(self.endpoint_id, str) or not _ID.fullmatch(self.endpoint_id):
            raise ValueError("RUNPOD_ENDPOINT_ID must be a valid endpoint ID.")
        try:
            self.timeout_seconds = float(raw_timeout)
        except (TypeError, ValueError) as exc:
            raise ValueError("RUNPOD_TIMEOUT_SECONDS must be a positive number.") from None
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("RUNPOD_TIMEOUT_SECONDS must be a positive number.")
        self._base = f"https://api.runpod.ai/v2/{self.endpoint_id}"
        self._opener = opener if opener is not None else request.build_opener(_NoRedirect())
        self.last_job = None

    @staticmethod
    def _encode(payload):
        try:
            return json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError):
            raise ValueError("RunPod input must be JSON serializable.") from None

    def _request(self, method, suffix, payload=None, *, timeout=30,
                 max_payload_bytes=MAX_RUN_PAYLOAD_BYTES):
        url = f"{self._base}/{suffix}"
        body = None
        if payload is not None:
            body = self._encode(payload)
            if len(body) > max_payload_bytes:
                raise ValueError(
                    "Image exceeds RunPod's request limit after base64 encoding. "
                    "Configure external object storage for full-resolution transfer."
                )
        headers = {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        req = request.Request(url, data=body, headers=headers, method=method)
        try:
            with self._opener.open(req, timeout=timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except error.HTTPError as exc:
            code = exc.code if type(exc.code) is int else "unknown"
            failure = _TransientRunPodError if code in _TRANSIENT_HTTP_CODES else RunPodError
            raise failure(f"RunPod request failed (HTTP {code}).") from None
        except (error.URLError, OSError, TimeoutError):
            raise _TransientRunPodError("RunPod request failed due to a network error.") from None
        if len(raw) > MAX_RESPONSE_BYTES:
            raise RunPodError("RunPod response exceeded the supported size.")
        try:
            result = json.loads(raw)
        except (UnicodeError, ValueError):
            raise RunPodError("RunPod returned invalid JSON.") from None
        if not isinstance(result, dict):
            raise RunPodError("RunPod returned an invalid response.")
        return result

    def submit(self, job_input):
        """Submit once and return the job ID. A network ambiguity is never retried."""
        if not isinstance(job_input, dict):
            raise ValueError("RunPod input must be an object.")
        data = self._request("POST", "run", {"input": job_input},
                             timeout=min(30, self.timeout_seconds))
        job_id = data.get("id")
        if not isinstance(job_id, str) or not _ID.fullmatch(job_id):
            raise RunPodError("RunPod did not return a valid job ID.")
        if data.get("status") not in ("IN_QUEUE", "IN_PROGRESS", "COMPLETED"):
            raise RunPodError("RunPod did not accept the job.")
        return job_id

    def cancel(self, job_id):
        if not isinstance(job_id, str) or not _ID.fullmatch(job_id):
            raise ValueError("Invalid RunPod job ID.")
        self._request("POST", f"cancel/{job_id}", timeout=5)

    def wait(self, job_id, *, timeout_seconds=None):
        if not isinstance(job_id, str) or not _ID.fullmatch(job_id):
            raise ValueError("Invalid RunPod job ID.")
        completed = False
        try:
            remaining = self.timeout_seconds if timeout_seconds is None else timeout_seconds
            if not _number(remaining):
                raise ValueError("RunPod wait timeout must be nonnegative.")
            deadline = time.monotonic() + remaining
            failures = 0
            while True:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise RunPodError("RunPod job timed out; cancellation was requested.")
                try:
                    data = self._request("GET", f"status/{job_id}", timeout=min(30, left))
                except _TransientRunPodError:
                    failures += 1
                    if time.monotonic() >= deadline:
                        raise RunPodError("RunPod job timed out; cancellation was requested.") from None
                    if failures >= _MAX_STATUS_FAILURES:
                        raise
                    time.sleep(min(.5 * 2 ** (failures - 1), 4.0,
                                   max(0, deadline - time.monotonic())))
                    continue
                failures = 0
                if data.get("id") != job_id:
                    raise RunPodError("RunPod returned an unexpected job ID.")
                status = data.get("status")
                if status == "COMPLETED":
                    if time.monotonic() >= deadline:
                        raise RunPodError("RunPod job timed out; cancellation was requested.")
                    completed = True
                    return data
                if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
                    raise RunPodError(f"RunPod job ended with status {status}.")
                if status not in ("IN_QUEUE", "IN_PROGRESS"):
                    raise RunPodError("RunPod returned an unknown job status.")
                time.sleep(min(1.0, max(0, deadline - time.monotonic())))
        finally:
            if not completed:
                # The job may still be billable after a broken status response,
                # network failure, or caller interruption. Keep the original
                # exception if the best-effort cancellation also fails.
                try:
                    self.cancel(job_id)
                except BaseException:
                    pass

    def run(self, job_input):
        """Wait on the original request, then poll the same job if still pending."""
        started = time.monotonic()
        if not isinstance(job_input, dict):
            raise ValueError("RunPod input must be an object.")
        body_size = len(self._encode({"input": job_input}))
        if body_size > MAX_SYNC_PAYLOAD_BYTES:
            raise ValueError(
                "Image exceeds RunPod's 20 MB request limit after base64 encoding. "
                "Configure external object storage for full-resolution transfer."
            )
        remaining = self.timeout_seconds - (time.monotonic() - started)
        if remaining < 1:
            raise ValueError("RUNPOD_TIMEOUT_SECONDS must allow at least 1 second for /runsync.")
        # Leave transport time to receive a pending job ID before the deadline.
        wait_ms = max(1000, int(min(300, remaining * .95) * 1000))
        response = self._request("POST", f"runsync?wait={wait_ms}", {"input": job_input},
                                 timeout=min(305, remaining),
                                 max_payload_bytes=MAX_SYNC_PAYLOAD_BYTES)
        job_id = response.get("id")
        if not isinstance(job_id, str) or not _ID.fullmatch(job_id):
            raise RunPodError("RunPod did not return a valid job ID.")
        self.last_job = {"job_id": job_id}
        pending = response.get("status") in ("IN_QUEUE", "IN_PROGRESS")
        try:
            if pending:
                response = self.wait(job_id, timeout_seconds=max(0, self.timeout_seconds -
                                                                 (time.monotonic() - started)))
            elif response.get("status") != "COMPLETED":
                raise RunPodError("RunPod job did not complete successfully.")
            elif time.monotonic() - started >= self.timeout_seconds:
                raise RunPodError("RunPod job timed out; cancellation was requested.")
        except BaseException:
            if not pending:
                try:
                    self.cancel(job_id)
                except BaseException:
                    pass
            raise
        finally:
            self.last_job["wall_ms"] = round((time.monotonic() - started) * 1000, 2)
        for key in ("delayTime", "executionTime"):
            if _number(response.get(key)):
                self.last_job[key] = response[key]
        output = response.get("output")
        if isinstance(output, dict):
            self.last_job["worker_metrics"] = _safe_metrics(output.get("metrics"), api_key=self.api_key)
        return response


class RunPodInferenceEngine:
    """CountGD++ per-image engine matching the application's inference API."""

    def __init__(self, api_key=None, endpoint_id=None, timeout_seconds=None, *, client=None):
        self.client = client if client is not None else RunPodClient(api_key, endpoint_id, timeout_seconds)
        self.last_jobs = []

    def close(self):
        pass

    def _infer(self, action, image_path, *, boxes=None, confidence=None, label=""):
        from .inference import _read_image

        image = _read_image(image_path)
        try:
            raw = Path(image_path).read_bytes()
            payload = {"action": action, "image_base64": base64.b64encode(raw).decode("ascii"),
                       "label": label}
            if action == "count":
                payload.update(boxes=boxes, confidence=confidence)
            response = self.client.run(payload)
            if isinstance(self.client.last_job, dict):
                self.last_jobs.append(dict(self.client.last_job))
            output = response.get("output")
            if not isinstance(output, dict) or not isinstance(output.get("result"), dict):
                raise RunPodError("RunPod worker returned an invalid result.")
            return image, output["result"]
        except BaseException:
            image.close()
            raise

    def suggest_reference_boxes(self, image_path: str, label: str = "") -> dict:
        image, result = self._infer("suggest", image_path, label=label)
        try:
            boxes = result.get("boxes")
            warning = result.get("warning")
            if (type(result.get("width")) is not int or result["width"] != image.width or
                    type(result.get("height")) is not int or result["height"] != image.height or
                    result.get("model") != "fastsam-s" or not _number(result.get("elapsed_ms")) or
                    not isinstance(boxes, list) or len(boxes) > MAX_DETECTIONS or
                    any(not _box(box, *image.size) for box in boxes) or
                    warning not in (None, _REFERENCE_WARNING)):
                raise RunPodError("RunPod worker returned invalid reference boxes.")
            return {"width": image.width, "height": image.height, "boxes": boxes,
                    "model": "fastsam-s", "elapsed_ms": result["elapsed_ms"],
                    "warning": warning}
        finally:
            image.close()

    def count(self, model_id: str, references: list[dict], image_paths: list[str],
              output_dir: str, confidence: float = .25, label: str = "", *,
              reference_strategy: str = "mean", reference_mode: str = "per_image") -> list[dict]:
        from .inference import _read_image

        if model_id != COUNTGD_MODEL_ID or reference_mode != "per_image" or reference_strategy != "mean":
            raise ValueError("Remote inference supports only CountGD++ per-image counting.")
        if not _number(confidence, maximum=1):
            raise ValueError("Confidence must be between 0 and 1.")
        if not isinstance(label, str):
            raise ValueError("Label must be text.")
        if not image_paths or not isinstance(references, list) or len(references) != len(image_paths):
            raise ValueError("CountGD++ needs one set of reference boxes for each image to count.")
        for reference, image_path in zip(references, image_paths, strict=True):
            if (not isinstance(reference, dict) or not isinstance(reference.get("path"), str) or
                    not Path(reference["path"]).is_absolute() or
                    Path(reference["path"]).resolve() != Path(image_path).resolve()):
                raise ValueError("Each CountGD++ reference path must match its target image.")
            image = _read_image(image_path)
            try:
                boxes = reference.get("boxes")
                if (not isinstance(boxes, list) or not 1 <= len(boxes) <= 32 or
                        any(not _box(box, *image.size) for box in boxes) or
                        len({tuple(box) for box in boxes}) != len(boxes)):
                    raise ValueError("Each CountGD++ target image needs 1–32 distinct in-bounds boxes.")
            finally:
                image.close()
        directory = Path(output_dir).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        self.last_jobs = []
        outputs = []
        for reference, image_path in zip(references, image_paths, strict=True):
            image, result = self._infer("count", image_path, boxes=reference["boxes"],
                                        confidence=confidence, label=label)
            try:
                outputs.append(self._count_result(result, image, directory))
            finally:
                image.close()
        return outputs

    @staticmethod
    def _count_result(result, image, directory):
        from .inference import _draw_detections

        count, detections = result.get("count"), result.get("detections")
        warning = result.get("warning")
        if ("annotated_path" in result or type(result.get("width")) is not int or
                result["width"] != image.width or type(result.get("height")) is not int or
                result["height"] != image.height or type(count) is not int or
                not 0 <= count <= MAX_DETECTIONS or not isinstance(detections, list) or
                len(detections) != count or not _number(result.get("elapsed_ms")) or
                warning not in (_WARNING, _WARNING + _LIMIT_WARNING)):
            raise RunPodError("RunPod worker returned an invalid count result.")
        checked = []
        for index, detection in enumerate(detections, start=1):
            if (not isinstance(detection, dict) or detection.get("index") != index or
                    type(detection.get("index")) is not int or
                    not _box(detection.get("box"), *image.size) or
                    not _number(detection.get("confidence"), maximum=1)):
                raise RunPodError("RunPod worker returned an invalid detection.")
            checked.append({"index": index, "box": detection["box"],
                            "confidence": detection["confidence"]})
        processing = result.get("processing")
        if processing is not None:
            limits = {"tiles": (1, 4096), "tile_size": (1, 16384),
                      "duplicates_removed": (0, 1_000_000)}
            if (not isinstance(processing, dict) or set(processing) != set(limits) or
                    any(type(processing[key]) is not int or not low <= processing[key] <= high
                        for key, (low, high) in limits.items())):
                raise RunPodError("RunPod worker returned invalid processing metadata.")
        destination = directory / f"{uuid4().hex}.jpg"
        _draw_detections(image, checked, destination)
        output = {"count": count, "detections": checked, "annotated_path": str(destination),
                  "width": image.width, "height": image.height,
                  "elapsed_ms": result["elapsed_ms"], "warning": warning}
        if processing is not None:
            output["processing"] = dict(processing)
        return output


def remote_model_catalog():
    """Catalog shape accepted by app.main._model_catalog, without loading weights."""
    endpoint = os.environ.get("RUNPOD_ENDPOINT_ID", "")
    available = bool(os.environ.get("RUNPOD_API_KEY", "").strip() and _ID.fullmatch(endpoint))
    return {"models": [{"id": COUNTGD_MODEL_ID, "name": "CountGD++", "family": "CountGD++",
                        "size": "base", "default_confidence": .80, "reference_mode": "per_image",
                        "required_reference_boxes": None, "recommended_reference_boxes": 10,
                        "available": available, "bytes": 0}],
            "default_model": COUNTGD_MODEL_ID, "device": "remote"}
