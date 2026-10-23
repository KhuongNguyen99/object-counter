"""Opt-in timing of the RunPod 1.12.0 queued result handoff."""

import contextvars
from functools import wraps
import hashlib
import json
import re
import time


_job_trace = contextvars.ContextVar("countwell_runpod_job_trace", default=None)


def _job_hash(job):
    job_id = job.get("id") if isinstance(job, dict) else None
    if not isinstance(job_id, str):
        return None
    try:
        return hashlib.sha256(job_id.encode("utf-8")).hexdigest()[:16]
    except UnicodeError:
        return None


def _duration_ms(start, end):
    return round(max(0, end - start) * 1000, 2) if start is not None and end is not None else None


def _error_type(exc):
    name = type(exc).__name__
    return name if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", name) else "OtherError"


def _emit(trace):
    try:
        handoff_ms = _duration_ms(trace.get("result_started"), trace.get("result_ended"))
        post_ms = _duration_ms(trace.get("post_started"), trace.get("post_ended"))
        pre_post_ms = _duration_ms(trace.get("result_started"), trace.get("post_started"))
        # Includes JSON encoding plus SDK setup; the SDK does not expose encoding alone.
        record = {
            "job_id_hash": trace["job_id_hash"],
            "handler_ms": _duration_ms(trace.get("handler_started"), trace.get("handler_ended")),
            "handler_status": trace.get("handler_status"),
            "handler_error_type": trace.get("handler_error_type"),
            "handler_to_result_ms": _duration_ms(trace.get("handler_ended"), trace.get("result_started")),
            "sdk_result_handoff_ms": handoff_ms,
            "sdk_result_pre_post_ms": pre_post_ms,
            "sdk_result_post_ms": post_ms,
            "sdk_result_after_post_ms": (
                round(max(0, handoff_ms - pre_post_ms - post_ms), 2)
                if None not in (handoff_ms, pre_post_ms, post_ms) else None
            ),
            "sdk_result_status": trace.get("result_status"),
            "sdk_result_error_type": trace.get("result_error_type"),
            "post_status": trace.get("post_status", "not_started"),
            "post_error_type": trace.get("post_error_type"),
        }
        print("COUNTWELL_WORKER_TIMING " + json.dumps(record, separators=(",", ":")), flush=True)
    except Exception:
        # Observability must never change result delivery.
        pass


def install(handler, rp_job, rp_http):
    """Instrument the pinned SDK's imported send_result and _transmit calls."""
    if not getattr(rp_job.send_result, "_countwell_timing_wrapped", False):
        original_send_result = rp_job.send_result

        @wraps(original_send_result)
        async def timed_send_result(*args, **kwargs):
            job = args[2] if len(args) > 2 else kwargs.get("job")
            trace = _job_trace.get()
            if trace is None or trace["job_id_hash"] != _job_hash(job):
                trace = {"job_id_hash": _job_hash(job)}
                _job_trace.set(trace)
            trace["result_started"] = time.perf_counter()
            trace["result_status"] = "returned"
            try:
                return await original_send_result(*args, **kwargs)
            except BaseException as exc:
                trace["result_status"] = "raised"
                trace["result_error_type"] = _error_type(exc)
                raise
            finally:
                trace["result_ended"] = time.perf_counter()
                _emit(trace)
                _job_trace.set(None)

        timed_send_result._countwell_timing_wrapped = True
        rp_job.send_result = timed_send_result

    if not getattr(rp_http._transmit, "_countwell_timing_wrapped", False):
        original_transmit = rp_http._transmit

        @wraps(original_transmit)
        async def timed_transmit(*args, **kwargs):
            trace = _job_trace.get()
            if trace is not None:
                trace["post_started"] = time.perf_counter()
                trace["post_status"] = "completed"
            try:
                return await original_transmit(*args, **kwargs)
            except BaseException as exc:
                if trace is not None:
                    trace["post_status"] = "raised"
                    trace["post_error_type"] = _error_type(exc)
                raise
            finally:
                if trace is not None:
                    trace["post_ended"] = time.perf_counter()

        timed_transmit._countwell_timing_wrapped = True
        rp_http._transmit = timed_transmit

    @wraps(handler)
    def timed_handler(event):
        trace = {"job_id_hash": _job_hash(event), "handler_started": time.perf_counter()}
        _job_trace.set(trace)
        trace["handler_status"] = "returned"
        try:
            return handler(event)
        except BaseException as exc:
            trace["handler_status"] = "raised"
            trace["handler_error_type"] = _error_type(exc)
            raise
        finally:
            trace["handler_ended"] = time.perf_counter()

    return timed_handler
