"""Local launcher tests; none of these start Docker or download assets."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import run as launcher


CONFIG = {
    "services": {
        "frontend": {
            "image": "countwell-frontend:local",
            "ports": [{"target": 8080, "published": "9187", "host_ip": "127.0.0.1"}],
        },
        "counter": {"image": "yoloe-counter:local"},
    },
}


@pytest.fixture(autouse=True)
def isolate_dotenv(monkeypatch):
    # Launcher tests must never read the private checkout's .env file.
    monkeypatch.setattr(launcher, "load_dotenv", lambda path, environ: False)
    monkeypatch.delenv("INFERENCE_BACKEND", raising=False)


def install_fake_docker(monkeypatch, *, version="2.40.0", os_type="linux",
                        failure=None, cuda_code=0, config=CONFIG):
    calls = []
    monkeypatch.setattr(launcher.shutil, "which", lambda name: "docker" if name == "docker" else None)

    def fake_docker(*args, **kwargs):
        calls.append((args, kwargs))
        if failure in args:
            raise subprocess.CalledProcessError(17, ["docker", *args], stderr="simulated failure")
        if args[:2] == ("compose", "version"):
            stdout = version
        elif args[:1] == ("info",):
            stdout = os_type
        elif "config" in args:
            stdout = json.dumps(config)
        elif "-c" in args:
            return subprocess.CompletedProcess(["docker", *args], cuda_code)
        else:
            stdout = ""
        return subprocess.CompletedProcess(["docker", *args], 0, stdout=stdout)

    monkeypatch.setattr(launcher, "docker", fake_docker)
    return calls


def compose_calls(calls, action):
    return [(args, kwargs) for args, kwargs in calls
            if args[:1] == ("compose",) and action in args]


def test_automatic_cpu_when_gpu_probe_reports_unavailable(monkeypatch, capsys):
    calls = install_fake_docker(monkeypatch)
    probes = []
    monkeypatch.setattr(launcher, "gpu_available", lambda: probes.append(True) or False)
    monkeypatch.setattr(launcher, "prepare_assets", lambda: None)
    monkeypatch.setenv("DEVICE", "0")

    assert launcher.main([]) == 0

    assert probes == [True]
    assert "Using CPU" in capsys.readouterr().out
    assert all("compose.gpu.yaml" not in args for args, _ in compose_calls(calls, "config"))
    assert all(kwargs["env"]["DEVICE"] == "cpu"
               for _, kwargs in compose_calls(calls, "config") +
               compose_calls(calls, "build") + compose_calls(calls, "up"))


@pytest.mark.parametrize("returncode,stdout", [(1, "GPU 0: present"), (0, "No devices were found")])
def test_gpu_probe_requires_success_and_a_listed_gpu(monkeypatch, returncode, stdout):
    monkeypatch.setattr(launcher, "image_exists", lambda image: True)

    def fake_docker(*args, **kwargs):
        assert "--gpus" in args
        return subprocess.CompletedProcess(["docker", *args], returncode, stdout=stdout)

    monkeypatch.setattr(launcher, "docker", fake_docker)
    assert launcher.gpu_available() is False


def test_timed_out_probe_removes_only_its_named_container(monkeypatch):
    monkeypatch.setattr(launcher.uuid, "uuid4", lambda: SimpleNamespace(hex="fixedprobe"))
    calls = []

    def fake_docker(*args, **kwargs):
        calls.append(args)
        if args[0] == "run":
            raise subprocess.TimeoutExpired(["docker", *args], 60)
        return subprocess.CompletedProcess(["docker", *args], 0)

    monkeypatch.setattr(launcher, "docker", fake_docker)
    with pytest.raises(subprocess.TimeoutExpired):
        launcher.run_probe("probe:image", "nvidia-smi", "-L")

    assert "--name" in calls[0]
    assert calls[0][calls[0].index("--name") + 1] == "countwell-check-fixedprobe"
    assert calls[1] == ("rm", "--force", "countwell-check-fixedprobe")
    assert len(calls) == 2


def test_forced_cpu_skips_gpu_probe_and_overrides_stale_device(monkeypatch):
    calls = install_fake_docker(monkeypatch)
    monkeypatch.setenv("DEVICE", "0")
    monkeypatch.setattr(launcher, "gpu_available", lambda: pytest.fail("GPU probe must be skipped"))
    monkeypatch.setattr(launcher, "prepare_assets", lambda: None)

    assert launcher.main(["--cpu"]) == 0
    assert all(kwargs["env"]["DEVICE"] == "cpu"
               for _, kwargs in compose_calls(calls, "config") +
               compose_calls(calls, "build") + compose_calls(calls, "up"))
    assert not [args for args, _ in calls if args[:1] == ("run",)]


def test_runpod_mode_uses_dotenv_copy_and_lightweight_overlay(monkeypatch, capsys):
    calls = install_fake_docker(monkeypatch)
    secret = "private-test-key"
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.delenv("RUNPOD_ENDPOINT_ID", raising=False)
    monkeypatch.setattr(launcher, "load_dotenv", lambda path, env: env.update({
        "INFERENCE_BACKEND": "runpod", "RUNPOD_API_KEY": secret,
        "RUNPOD_ENDPOINT_ID": "endpoint-123",
    }))
    monkeypatch.setattr(launcher, "gpu_available", lambda: pytest.fail("GPU probe must be skipped"))
    monkeypatch.setattr(launcher, "prepare_assets", lambda: pytest.fail("assets must be skipped"))
    monkeypatch.setattr(launcher, "run_probe", lambda *args, **kwargs: pytest.fail("CUDA probe must be skipped"))

    assert launcher.main([]) == 0
    assert "RUNPOD_API_KEY" not in launcher.os.environ
    assert all("compose.runpod.yaml" in args and "compose.gpu.yaml" not in args
               for args, _ in compose_calls(calls, "config") +
               compose_calls(calls, "build") + compose_calls(calls, "up"))
    assert all(kwargs["env"]["RUNPOD_API_KEY"] == secret and
               kwargs["env"]["INFERENCE_BACKEND"] == "runpod"
               for _, kwargs in compose_calls(calls, "config") +
               compose_calls(calls, "build") + compose_calls(calls, "up"))
    captured = capsys.readouterr()
    assert "RunPod Serverless" in captured.out
    assert secret not in captured.out + captured.err


def test_explicit_local_mode_overrides_dotenv_runpod(monkeypatch):
    calls = install_fake_docker(monkeypatch)
    monkeypatch.setattr(launcher, "load_dotenv", lambda path, env: env.update({
        "INFERENCE_BACKEND": "runpod",
    }))
    monkeypatch.setattr(launcher, "gpu_available", lambda: pytest.fail("GPU probe must be skipped"))
    monkeypatch.setattr(launcher, "prepare_assets", lambda: None)

    assert launcher.main(["--cpu"]) == 0
    assert all("compose.runpod.yaml" not in args and kwargs["env"]["INFERENCE_BACKEND"] == "local"
               for args, kwargs in compose_calls(calls, "config") +
               compose_calls(calls, "build") + compose_calls(calls, "up"))


def test_explicit_gpu_overrides_dotenv_runpod(monkeypatch):
    calls = install_fake_docker(monkeypatch)
    monkeypatch.setattr(launcher, "load_dotenv", lambda path, env: env.update({
        "INFERENCE_BACKEND": "runpod",
    }))
    monkeypatch.setattr(launcher, "gpu_available", lambda: True)
    monkeypatch.setattr(launcher, "prepare_assets", lambda: None)

    assert launcher.main(["--gpu"]) == 0
    assert all("compose.gpu.yaml" in args and "compose.runpod.yaml" not in args and
               kwargs["env"]["INFERENCE_BACKEND"] == "local"
               for args, kwargs in compose_calls(calls, "config") +
               compose_calls(calls, "build") + compose_calls(calls, "up"))


def test_explicit_runpod_overrides_dotenv_local(monkeypatch):
    calls = install_fake_docker(monkeypatch)
    monkeypatch.setattr(launcher, "load_dotenv", lambda path, env: env.update({
        "INFERENCE_BACKEND": "local", "RUNPOD_API_KEY": "test-key",
        "RUNPOD_ENDPOINT_ID": "endpoint-123",
    }))
    monkeypatch.setattr(launcher, "gpu_available", lambda: pytest.fail("GPU probe must be skipped"))
    monkeypatch.setattr(launcher, "prepare_assets", lambda: pytest.fail("assets must be skipped"))

    assert launcher.main(["--runpod"]) == 0
    assert all("compose.runpod.yaml" in args and kwargs["env"]["INFERENCE_BACKEND"] == "runpod"
               for args, kwargs in compose_calls(calls, "config") +
               compose_calls(calls, "build") + compose_calls(calls, "up"))


def test_explicit_runpod_requires_credentials_without_exposing_key(monkeypatch, capsys):
    calls = install_fake_docker(monkeypatch)
    secret = "private-test-key"
    monkeypatch.setattr(launcher, "load_dotenv", lambda path, env: env.update({
        "RUNPOD_API_KEY": secret, "RUNPOD_ENDPOINT_ID": "bad endpoint",
    }))
    assert launcher.main(["--runpod"]) == 1
    captured = capsys.readouterr()
    assert "RUNPOD_ENDPOINT_ID" in captured.err
    assert secret not in captured.out + captured.err
    assert not compose_calls(calls, "config")


def test_runpod_flag_is_mutually_exclusive_with_local_flags():
    with pytest.raises(SystemExit) as exc:
        launcher.main(["--runpod", "--gpu"])
    assert exc.value.code == 2


def test_detected_gpu_selects_overlay_and_checks_cuda(monkeypatch):
    calls = install_fake_docker(monkeypatch)
    monkeypatch.setattr(launcher, "gpu_available", lambda: True)
    monkeypatch.setattr(launcher, "prepare_assets", lambda: None)
    monkeypatch.setenv("DEVICE", "cpu")

    assert launcher.main([]) == 0

    assert all("compose.gpu.yaml" in args for args, _ in
               compose_calls(calls, "config") + compose_calls(calls, "build") +
               compose_calls(calls, "up"))
    cuda = [args for args, _ in calls if args[:1] == ("run",) and "-c" in args]
    assert len(cuda) == 1
    assert CONFIG["services"]["counter"]["image"] in cuda[0]
    assert "--gpus" in cuda[0]
    assert "torch.cuda.synchronize()" in cuda[0][-1]


def test_explicit_gpu_requirement_fails_without_retry(monkeypatch, capsys):
    calls = install_fake_docker(monkeypatch)
    monkeypatch.setattr(launcher, "gpu_available", lambda: False)
    monkeypatch.setattr(launcher, "prepare_assets", lambda: pytest.fail("must not prepare assets"))

    assert launcher.main(["--gpu"]) == 1

    captured = capsys.readouterr()
    assert "NVIDIA GPU" in captured.err
    assert "ready" not in captured.out.lower()
    assert not compose_calls(calls, "config")
    assert not compose_calls(calls, "up")


@pytest.mark.parametrize("failure,mode", [("build", ["--cpu"]), ("up", ["--cpu"]),
                                           ("cuda", ["--gpu"])])
def test_startup_failure_reports_nonzero_without_false_ready_or_cpu_retry(
        monkeypatch, capsys, failure, mode):
    calls = install_fake_docker(monkeypatch, failure=failure if failure != "cuda" else None,
                                cuda_code=1 if failure == "cuda" else 0)
    monkeypatch.setattr(launcher, "gpu_available", lambda: True)
    monkeypatch.setattr(launcher, "prepare_assets", lambda: None)

    assert launcher.main(mode) == 1

    captured = capsys.readouterr()
    assert "Could not start Countwell" in captured.err
    assert "Countwell is ready" not in captured.out
    assert len(compose_calls(calls, "build")) == 1
    assert len(compose_calls(calls, "up")) == (1 if failure == "up" else 0)
    if failure == "cuda":
        assert all("compose.gpu.yaml" in args for args, _ in compose_calls(calls, "build"))


@pytest.mark.parametrize("cached", [True, False])
def test_no_build_requires_cached_images_and_skips_preparation(monkeypatch, capsys, cached):
    calls = install_fake_docker(monkeypatch)
    monkeypatch.setattr(launcher, "gpu_available", lambda: False)
    monkeypatch.setattr(launcher, "prepare_assets", lambda: pytest.fail("assets must be skipped"))
    inspected = []

    def image_exists(image):
        inspected.append(image)
        return cached

    monkeypatch.setattr(launcher, "image_exists", image_exists)
    assert launcher.main(["--cpu", "--no-build"]) == (0 if cached else 1)
    assert inspected == (["countwell-frontend:local", "yoloe-counter:local"] if cached
                         else ["countwell-frontend:local"])
    assert not compose_calls(calls, "build")
    assert bool(compose_calls(calls, "up")) is cached
    if not cached:
        assert "Countwell is ready" not in capsys.readouterr().out


def test_default_builds_then_starts_both_services_and_waits(monkeypatch, capsys):
    calls = install_fake_docker(monkeypatch)
    monkeypatch.setattr(launcher, "gpu_available", lambda: False)
    prepared = []
    monkeypatch.setattr(launcher, "prepare_assets", lambda: prepared.append(True))

    assert launcher.main([]) == 0

    assert prepared == [True]
    build = compose_calls(calls, "build")
    up = compose_calls(calls, "up")
    assert len(build) == len(up) == 1
    assert calls.index(build[0]) < calls.index(up[0])
    assert {"--detach", "--no-build", "--wait", "--pull", "never"}.issubset(up[0][0])
    assert "http://127.0.0.1:9187" in capsys.readouterr().out


@pytest.mark.parametrize("host,expected", [
    ("0.0.0.0", "http://localhost:9187"),
    ("::", "http://localhost:9187"),
    ("::1", "http://[::1]:9187"),
    ("192.0.2.12", "http://192.0.2.12:9187"),
])
def test_service_url_uses_configured_published_port(host, expected):
    assert launcher.service_url({"ports": [
        {"target": 8082, "published": "9000", "host_ip": host},
        {"target": 8080, "published": "9187", "host_ip": host},
    ]}) == expected


@pytest.mark.parametrize("path_argument", [
    r"C:\Program Files\Countwell\compose file.yaml",
    "/srv/countwell files/compose file.yaml",
])
def test_docker_uses_script_root_and_argument_list(monkeypatch, tmp_path, path_argument):
    root = tmp_path / "repo with spaces"
    root.mkdir()
    monkeypatch.setattr(launcher, "ROOT", root)
    recorded = []

    def fake_run(command, **kwargs):
        recorded.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    launcher.docker("compose", "-f", path_argument, "config")

    command, kwargs = recorded[0]
    assert command == ["docker", "compose", "-f", path_argument, "config"]
    assert kwargs["cwd"] == root
    assert kwargs["shell"] is False


def test_prepare_assets_uses_current_python_and_absolute_paths(monkeypatch, tmp_path):
    root = tmp_path / "repo with spaces"
    root.mkdir()
    monkeypatch.setattr(launcher, "ROOT", root)
    recorded = []

    def fake_run(command, **kwargs):
        recorded.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    launcher.prepare_assets()

    command, kwargs = recorded[0]
    assert (root / "models" / "countgd").is_dir()
    assert command[0] == sys.executable
    assert Path(command[1]) == root / "scripts" / "download_models.py"
    assert Path(command[command.index("--model-dir") + 1]) == root / "models"
    assert Path(command[1]).is_absolute()
    assert kwargs["cwd"] == root and kwargs["shell"] is False


def test_missing_docker_fails_before_any_command(monkeypatch, capsys):
    monkeypatch.setattr(launcher.shutil, "which", lambda name: None)
    monkeypatch.setattr(launcher, "docker", lambda *args, **kwargs: pytest.fail("Docker must not run"))

    assert launcher.main([]) == 1
    assert "Docker was not found" in capsys.readouterr().err


@pytest.mark.parametrize("version,os_type,expected", [
    ("2.29.9", "linux", "2.30"),
    ("2.40.0", "windows", "Linux containers"),
])
def test_unsupported_docker_environment_fails_before_build(
        monkeypatch, capsys, version, os_type, expected):
    calls = install_fake_docker(monkeypatch, version=version, os_type=os_type)
    monkeypatch.setattr(launcher, "gpu_available", lambda: pytest.fail("GPU probe must not run"))
    monkeypatch.setattr(launcher, "prepare_assets", lambda: pytest.fail("assets must not prepare"))

    assert launcher.main([]) == 1
    assert expected in capsys.readouterr().err
    assert not compose_calls(calls, "config")
    assert not compose_calls(calls, "up")


def test_runpod_can_read_an_alternate_private_env_file(monkeypatch, tmp_path):
    calls = install_fake_docker(monkeypatch)
    settings = tmp_path / ".env.runpod"
    seen = []

    def private_settings(path, env):
        seen.append(path)
        env.update(RUNPOD_API_KEY="test-key", RUNPOD_ENDPOINT_ID="endpoint-123")

    monkeypatch.setattr(launcher, "load_dotenv", private_settings)
    monkeypatch.setattr(launcher, "gpu_available", lambda: pytest.fail("No local GPU probe"))
    monkeypatch.setattr(launcher, "prepare_assets", lambda: pytest.fail("No model downloads"))
    assert launcher.main(["--runpod", "--env-file", str(settings)]) == 0
    assert seen == [settings]
    assert compose_calls(calls, "up")
