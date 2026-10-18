"""Keep PET-DINO's older MMDetection dependencies in its bundled Python runtime."""

import json
import os
from pathlib import Path
import queue
import subprocess
import threading

from .models import validate_petdino_assets


class PetDINOEngine:
    def __init__(self, model_dir: Path, device: str = "auto") -> None:
        validate_petdino_assets(model_dir)
        executable = Path(os.environ.get("PETDINO_PYTHON", "/opt/petdino-venv/bin/python"))
        if not executable.is_file():
            raise FileNotFoundError("PET-DINO's bundled Python runtime is unavailable.")
        self._responses = queue.Queue()
        self._process = subprocess.Popen(
            [str(executable), "-u", "-m", "app.petdino_runtime", "--model-dir", str(model_dir), "--device", device],
            cwd=Path(__file__).resolve().parents[1], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1,
        )
        self._reader = threading.Thread(target=self._read_output, name="petdino-output", daemon=True)
        self._reader.start()
        try:
            ready = self._receive(180)
            if ready.get("ready") is not True:
                raise RuntimeError(ready.get("error") or "PET-DINO could not start.")
        except Exception:
            self.close()
            raise

    def _read_output(self):
        try:
            for line in self._process.stdout:
                self._responses.put(line)
        except (OSError, ValueError):
            pass
        finally:
            self._responses.put(None)

    def _receive(self, timeout):
        try:
            line = self._responses.get(timeout=timeout)
        except queue.Empty as exc:
            self.close()
            raise RuntimeError("PET-DINO stopped responding; retry the batch.") from exc
        if line is None:
            raise RuntimeError("PET-DINO's worker exited; retry the batch.")
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("Response is not an object")
            return value
        except (ValueError, TypeError) as exc:
            self.close()
            raise RuntimeError("PET-DINO returned an invalid worker response.") from exc

    def count(self, references, image_paths, output_dir, confidence=.3, label=""):
        try:
            if self._process.poll() is not None:
                raise RuntimeError("PET-DINO's worker exited; retry the batch.")
            message = {"action": "count", "references": references, "image_paths": image_paths,
                       "output_dir": output_dir, "confidence": confidence, "label": label}
            self._process.stdin.write(json.dumps(message, allow_nan=False) + "\n")
            self._process.stdin.flush()
            response = self._receive(1800)
            if response.get("ok") is not True:
                raise RuntimeError(response.get("error") or "PET-DINO counting failed.")
            results = response.get("results")
            if not isinstance(results, list) or len(results) != len(image_paths):
                raise RuntimeError("PET-DINO returned the wrong number of image results.")
            return results
        except (BrokenPipeError, OSError) as exc:
            self.close()
            raise RuntimeError("PET-DINO's worker disconnected; retry the batch.") from exc

    def close(self):
        process = self._process
        if process.poll() is None:
            try:
                process.stdin.write('{"action":"close"}\n')
                process.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        for pipe in (process.stdin, process.stdout):
            if pipe is not None:
                try:
                    pipe.close()
                except (OSError, ValueError):
                    pass
        self._reader.join(timeout=1)
