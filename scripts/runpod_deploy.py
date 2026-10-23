"""Create and clean up owned, scale-to-zero RunPod Serverless endpoints.

Uses RunPod REST v2 (https://api.runpod.io/v2/openapi.json). Creating an
endpoint requires an already published container image. This script never
uploads source, publishes an image, creates a registry credential, or funds
an account. Secrets are read by the CLI at runtime and never saved here.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from urllib import error, parse, request
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

API_BASE = "https://api.runpod.io/v2"
ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class ApiError(RuntimeError):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class RunPodApi:
    """Fixed-host management client; failed creates are deliberately not retried."""

    def __init__(self, api_key, *, opener=None):
        if not api_key or not isinstance(api_key, str):
            raise ValueError("RUNPOD_API_KEY is required.")
        self.api_key = api_key
        self.opener = opener or request.build_opener(_NoRedirect())

    def request(self, method, path, data=None, *, timeout=20):
        if not path.startswith("/") or path.startswith("//"):
            raise ValueError("Expected a RunPod API path.")
        body = None if data is None else json.dumps(data, allow_nan=False).encode()
        req = request.Request(API_BASE + path, data=body, method=method,
                              headers={"Authorization": "Bearer " + self.api_key,
                                       "Content-Type": "application/json", "Accept": "application/json"})
        try:
            with self.opener.open(req, timeout=timeout) as response:
                raw = response.read(8_000_001)
        except error.HTTPError as exc:
            raise ApiError(f"RunPod management request failed (HTTP {exc.code}).", exc.code) from None
        except (error.URLError, OSError, TimeoutError):
            raise ApiError("RunPod management request failed due to a network error.") from None
        if not raw:
            return {}
        if len(raw) > 8_000_000:
            raise ApiError("RunPod management response exceeds the size limit.")
        try:
            value = json.loads(raw)
        except (UnicodeError, ValueError):
            raise ApiError("RunPod management response is not JSON.") from None
        if not isinstance(value, dict):
            raise ApiError("RunPod management response is not an object.")
        return value

    def catalog(self):
        result = self.request("GET", "/catalog/gpus?include=AVAILABILITY&product=SERVERLESS")
        if not isinstance(result.get("gpus"), list) or not result["gpus"]:
            raise ApiError("RunPod returned an empty GPU catalog.")
        return result["gpus"]


def normalize_gpu(name):
    # CUDA reports e.g. NVIDIA RTX A5000, while the catalog display name is RTX A5000.
    return re.sub(r"[^a-z0-9]", "", str(name).lower().removeprefix("nvidia "))


def select_gpu(catalog, name):
    matches = [gpu for gpu in catalog if normalize_gpu(name) in
               (normalize_gpu(gpu.get("id")), normalize_gpu(gpu.get("name")))]
    if len(matches) != 1:
        raise ValueError(f"GPU selection {name!r} did not identify one catalog card.")
    gpu = matches[0]
    price = gpu.get("price", {}).get("serverless")
    if (not gpu.get("pool") or type(price) not in (int, float) or
            not math.isfinite(price) or price <= 0):
        raise ValueError(f"{name} has no serverless pool and positive hourly quote.")
    excluded = sorted(card["id"] for card in catalog
                      if card.get("pool") == gpu["pool"] and card["id"] != gpu["id"])
    return {"id": gpu["id"], "name": gpu["name"], "pool": gpu["pool"],
            "excluded_types": excluded, "hourly_usd": price,
            "availability": gpu.get("availability"), "quoted_at": utc_now()}


def endpoint_body(name, image, gpu, *, timeout_seconds=300, idle_seconds=30,
                  disk_gb=30, env=None):
    if not image or not isinstance(image, str):
        raise ValueError("A published container image is required.")
    if not 1 <= timeout_seconds <= 1800 or not 1 <= idle_seconds <= 60 or not 1 <= disk_gb <= 100:
        raise ValueError("Timeout, idle timeout, or disk size exceeds the supported benchmark limits.")
    return {"name": name, "image": image, "type": "QUEUE", "disk": disk_gb,
            "gpu": {"pools": [gpu["pool"]], "excludedTypes": gpu["excluded_types"], "count": 1},
            "workers": {"min": 0, "max": 1, "idleTimeout": idle_seconds},
            "scaling": {"type": "QUEUE_DELAY", "queueDelay": 1},
            "timeout": int(timeout_seconds * 1000), "flashboot": "OFF",
            "env": dict(env or {})}


class OwnedEndpoints:
    """Persist creation intents before mutations; never operate on arbitrary IDs."""

    def __init__(self, api, path):
        self.api, self.path = api, Path(path)
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            if self.data.get("schema") != 1 or not isinstance(self.data.get("resources"), list):
                raise ValueError("Invalid RunPod ownership manifest.")
        else:
            self.data = {"schema": 1, "run_id": uuid4().hex, "created_at": utc_now(), "resources": []}
            self.save()

    def save(self):
        save_json(self.path, self.data)

    def create(self, image, gpu, **kwargs):
        ordinal = len(self.data["resources"]) + 1
        name = f"object-count-bench-{self.data['run_id']}-{ordinal}"
        body = endpoint_body(name, image, gpu, **kwargs)
        # Never persist env values: a caller may use a credential in the worker environment.
        record = {"name": name, "image": image, "gpu": gpu, "id": None,
                  "state": "creating", "created_at": utc_now()}
        self.data["resources"].append(record)
        self.save()
        try:
            endpoint = self.api.request("POST", "/serverless", body)
        except ApiError as exc:
            if exc.status is not None and 400 <= exc.status < 500 and exc.status != 408:
                record.update(state="rejected", rejected_at=utc_now())
                self.save()
            raise
        endpoint_id = endpoint.get("id")
        if not isinstance(endpoint_id, str) or not ID.fullmatch(endpoint_id):
            raise ApiError("RunPod create did not return an endpoint ID; run cleanup to reconcile the intent.")
        record.update(id=endpoint_id, state="created")
        self.save()
        live = self.api.request("GET", "/serverless/" + endpoint_id)
        expected_gpu = body["gpu"]
        actual_gpu = live.get("gpu", {})
        if (live.get("image") != image or live.get("type") != "QUEUE" or
                live.get("workers") != body["workers"] or live.get("timeout") != body["timeout"] or
                actual_gpu.get("count") != 1 or actual_gpu.get("pools") != expected_gpu["pools"] or
                sorted(actual_gpu.get("excludedTypes", [])) != sorted(expected_gpu["excludedTypes"])):
            raise ApiError("Created endpoint settings differ from the requested GPU and spending safeguards.")
        return record

    def reconcile(self, record):
        if record.get("id") or record.get("state") in ("deleted", "rejected"):
            return
        cursor, seen, matches = None, set(), []
        for _ in range(100):
            query = "?limit=1000" + ("&cursor=" + parse.quote(cursor, safe="") if cursor else "")
            page = self.api.request("GET", "/serverless" + query)
            matches.extend(item for item in page.get("endpoints", [])
                           if item.get("name") == record["name"] and item.get("image") == record["image"])
            paging = page.get("pagination", {})
            if not paging.get("hasNextPage"):
                break
            cursor = paging.get("nextCursor")
            if not cursor or cursor in seen:
                raise ApiError("Cannot reconcile endpoint creation: invalid pagination.")
            seen.add(cursor)
        else:
            raise ApiError("Cannot reconcile endpoint creation: page limit reached.")
        if len(matches) > 1:
            raise ApiError("Multiple endpoints match a single creation intent; manual review is required.")
        if matches:
            record.update(id=matches[0]["id"], state="created")
            self.save()
        # Keep unmatched intent for later reconciliation: a timed-out create may complete later.

    def delete(self, record):
        if record.get("state") in ("deleted", "rejected"):
            return
        self.reconcile(record)
        endpoint_id = record.get("id")
        if not endpoint_id:
            raise ApiError("Creation outcome is unresolved; rerun cleanup using the saved manifest.")
        if not ID.fullmatch(endpoint_id):
            raise ValueError("Invalid owned endpoint ID.")
        try:
            live = self.api.request("GET", "/serverless/" + endpoint_id)
            if live.get("name") != record["name"] or live.get("image") != record["image"]:
                raise ApiError("Endpoint no longer matches the ownership manifest; cleanup refused.")
            self.api.request("DELETE", "/serverless/" + endpoint_id)
        except ApiError as exc:
            if exc.status != 404:
                raise
        record.update(state="deleted", deleted_at=utc_now())
        self.save()

    def cleanup(self, *, keep_id=None):
        failures = []
        for record in self.data["resources"]:
            if record.get("id") == keep_id and keep_id is not None:
                continue
            for attempt in range(3):
                try:
                    self.delete(record)
                    break
                except (ApiError, OSError, ValueError) as exc:
                    if attempt == 2:
                        failures.append({"id": record.get("id"), "name": record["name"],
                                         "error": str(exc)})
                    else:
                        time.sleep(1)
        self.data["cleanup_errors"] = failures
        self.save()
        return failures


def load_api_key(env_file):
    from app.environment import load_dotenv
    load_dotenv(env_file)
    key = os.environ.get("RUNPOD_API_KEY", "")
    if not key:
        raise ValueError("Set RUNPOD_API_KEY in the environment or ignored .env file.")
    return key


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("catalog", help="Read current hardware availability and serverless price quotes")
    cleanup = commands.add_parser("cleanup", help="Delete only endpoints recorded by this tool")
    cleanup.add_argument("--manifest", required=True, type=Path)
    deploy = commands.add_parser("create", help="Create one scale-to-zero production endpoint")
    deploy.add_argument("--image", required=True)
    deploy.add_argument("--gpu", default="RTX A5000")
    deploy.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        api = RunPodApi(load_api_key(args.env_file))
        if args.command == "catalog":
            print(json.dumps(api.catalog(), indent=2))
        elif args.command == "cleanup":
            failures = OwnedEndpoints(api, args.manifest).cleanup()
            print(json.dumps({"cleanup_complete": not failures, "errors": failures}))
            return int(bool(failures))
        else:
            owned = OwnedEndpoints(api, args.manifest)
            try:
                resource = owned.create(args.image, select_gpu(api.catalog(), args.gpu), idle_seconds=5)
            except BaseException:
                owned.cleanup()
                raise
            print(json.dumps({"endpoint_id": resource["id"], "workers_min": 0, "workers_max": 1}))
    except (ApiError, ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
