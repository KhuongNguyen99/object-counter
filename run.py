#!/usr/bin/env python3
"""Start Countwell's frontend and model backend on Windows or Linux."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid

from app.environment import load_dotenv

ROOT = Path(__file__).resolve().parent
PROBE_IMAGE = "python:3.11-slim-bookworm"
CUDA_CHECK = (
    "import torch; x = torch.ones(1, device='cuda'); "
    "assert (x + x).item() == 2; torch.cuda.synchronize()"
)


def docker(*args, capture=False, check=True, env=None, timeout=None):
    return subprocess.run(
        ["docker", *args], cwd=ROOT, env=env, check=check, shell=False,
        capture_output=capture, text=True, encoding="utf-8", errors="replace",
        timeout=timeout,
    )


def image_exists(image):
    return docker("image", "inspect", image, capture=True, check=False,
                  timeout=30).returncode == 0


def run_probe(image, entrypoint, *args, capture=False):
    name = f"countwell-check-{uuid.uuid4().hex}"
    try:
        return docker("run", "--rm", "--name", name, "--network", "none", "--gpus", "all",
                      "--entrypoint", entrypoint, image, *args,
                      capture=capture, check=False, timeout=60)
    except (KeyboardInterrupt, subprocess.TimeoutExpired):
        # A stopped CLI can leave its container running on the Docker daemon.
        try:
            docker("rm", "--force", name, capture=True, check=False, timeout=10)
        except (OSError, subprocess.SubprocessError):
            pass
        raise


def gpu_available():
    print("Checking GPU access in Docker...", flush=True)
    image = next((name for name in ("yoloe-counter:cuda128", "yoloe-counter:local", PROBE_IMAGE)
                  if image_exists(name)), None)
    if image is None:
        docker("pull", PROBE_IMAGE)
        image = PROBE_IMAGE
    probe = run_probe(image, "nvidia-smi", "-L", capture=True)
    return probe.returncode == 0 and bool(re.search(r"GPU \d+:", probe.stdout))


def prepare_assets():
    # The backend build prepares the large counting assets using its own dependencies.
    (ROOT / "models" / "countgd").mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "download_models.py"),
         "--model-dir", str(ROOT / "models"), "--models", "FastSAM-s"],
        cwd=ROOT, check=True, shell=False,
    )


def service_url(service):
    port = next(port for port in service["ports"] if int(port["target"]) == 8080)
    host = port.get("host_ip", "0.0.0.0")
    if host in ("0.0.0.0", "::"):
        host = "localhost"
    if ":" in host:
        host = f"[{host}]"
    return f"http://{host}:{port['published']}"


def launch(args):
    if sys.version_info < (3, 11):
        raise RuntimeError("Install Python 3.11 or newer, then run this script again.")
    env = os.environ.copy()
    load_dotenv(ROOT / ".env", env)
    backend = "local" if args.cpu or args.gpu else ("runpod" if args.runpod else
              env.get("INFERENCE_BACKEND", "local").lower())
    if backend not in {"local", "runpod"}:
        raise ValueError("INFERENCE_BACKEND must be local or runpod.")
    env["INFERENCE_BACKEND"] = backend
    remote = backend == "runpod"
    if remote:
        from app.runpod_client import RunPodClient
        RunPodClient(api_key=env.get("RUNPOD_API_KEY", ""),
                     endpoint_id=env.get("RUNPOD_ENDPOINT_ID", ""),
                     timeout_seconds=env.get("RUNPOD_TIMEOUT_SECONDS", "600"))
    if not shutil.which("docker"):
        raise RuntimeError("Docker was not found. Install Docker and its Compose plugin first.")
    version = docker("compose", "version", "--short", capture=True, timeout=30).stdout.strip()
    match = re.match(r"v?(\d+)\.(\d+)", version)
    if not match or tuple(map(int, match.groups())) < (2, 30):
        raise RuntimeError("Docker Compose 2.30 or newer is required. Update Docker first.")
    if docker("info", "--format", "{{.OSType}}", capture=True, timeout=30).stdout.strip() != "linux":
        raise RuntimeError("Switch Docker to Linux containers, then run this script again.")

    use_gpu = not remote and not args.cpu and gpu_available()
    if args.gpu and not use_gpu:
        raise RuntimeError("Docker cannot access an NVIDIA GPU. Enable GPU access or run with --cpu.")
    print("Using RunPod Serverless." if remote else
          ("Using GPU." if use_gpu else "Using CPU (GPU is unavailable or --cpu was chosen)."),
          flush=True)
    if not remote and not use_gpu:
        env["DEVICE"] = "cpu"
    compose = ["compose", "-f", "compose.yaml"]
    if remote:
        compose += ["-f", "compose.runpod.yaml"]
    elif use_gpu:
        compose += ["-f", "compose.gpu.yaml"]
    config = json.loads(docker(*compose, "config", "--format", "json", capture=True,
                               env=env, timeout=30).stdout)
    if args.no_build:
        for service in config["services"].values():
            if not image_exists(service["image"]):
                raise RuntimeError("App images are missing. Run again without --no-build to build them.")
    else:
        print("Preparing the app. First-time downloads and builds can take a while...", flush=True)
        if not remote:
            prepare_assets()
        docker(*compose, "build", env=env)
    if use_gpu:
        print("Checking GPU counting support...", flush=True)
        probe = run_probe(config["services"]["counter"]["image"], "python", "-c", CUDA_CHECK)
        if probe.returncode:
            raise RuntimeError("The GPU counting check failed. Update the NVIDIA driver or run with --cpu.")

    print("Starting both services and waiting for them to be ready...", flush=True)
    docker(*compose, "up", "--detach", "--no-build", "--pull", "never",
           "--wait", "--wait-timeout", "120", env=env)
    print(f"\nCountwell is ready: {service_url(config['services']['frontend'])}")
    print("The app runs in the background. Saved counts stay in the existing data volume.")
    print("To stop it: docker compose stop")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--cpu", action="store_true", help="Use the CPU instead of checking for a GPU")
    mode.add_argument("--gpu", action="store_true", help="Require a working NVIDIA GPU in Docker")
    mode.add_argument("--runpod", action="store_true", help="Use the configured RunPod Serverless endpoint")
    parser.add_argument("--no-build", action="store_true", help="Start existing app images without building or preparing model assets")
    args = parser.parse_args(argv)
    try:
        launch(args)
        return 0
    except KeyboardInterrupt:
        print("\nStartup interrupted. Check containers with: docker compose ps", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"\nCould not start Countwell: {error}", file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            print(error.stderr.strip(), file=sys.stderr)
        print("Check that Docker is running and accessible. App logs: docker compose logs --tail 50", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
