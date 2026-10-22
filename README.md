# Countwell

Countwell counts objects in photos using a browser frontend and a local model backend. [CountGD++](https://github.com/niki-amini-naieni/CountGDPlusPlus) uses ten selected examples from each photo. [FastSAM-s](https://docs.ultralytics.com/models/fast-sam/) suggests boxes and powers Quick count within marked areas. Users can review results, correct counts, and export CSV.

The frontend and backend run in separate containers. Nginx serves the UI and proxies API requests to FastAPI, which owns inference, counting jobs, and saved data. Inference runs locally without a cloud API or API key.

## Run

You need Python 3.11+, Docker with Linux containers, and Docker Compose 2.30+. Use Docker Desktop with WSL2 on Windows. NVIDIA GPU support on Linux requires NVIDIA Container Toolkit.

From the repository root:

```powershell
# Windows
py -3 run.py
```

```sh
# Linux
python3 run.py
```

The launcher builds and starts both services, using a GPU when available and CPU otherwise. It waits until both services are healthy. The first build downloads the model assets; unchanged builds use Docker's cache. Add `--cpu` to force CPU mode, `--gpu` to require GPU support, or `--no-build` to reuse existing app images.

Open [localhost:8080](http://localhost:8080). Backend API docs are at [127.0.0.1:8082/docs](http://127.0.0.1:8082/docs).

Stop the app with `docker compose stop`. Saved counts, photos, and results stay in the `counter-data` volume at `/app/data`. `docker compose down -v` deletes that data.

## Repository structure

```text
frontend/          Browser UI, Nginx configuration, and frontend Dockerfile
app/main.py        FastAPI routes and counting queue
app/inference.py   Model inference
app/storage.py     SQLite sessions and saved results
scripts/           Model preparation and benchmarks
models/            Downloaded model assets (gitignored)
tests/             Python tests and browser checks
run.py             Windows and Linux launcher
Dockerfile         Backend image
compose*.yaml      Service configuration and GPU override
```

## Configuration

Copy [.env.example](.env.example) to `.env` to change ports, network bindings, and CPU threads. The frontend defaults to port 8080; the backend API is bound to localhost on port 8082.

For separate hosts, set `BACKEND_URL` to the backend origin reachable from the frontend container. To have browsers call the backend directly, set `API_BASE_URL` and list the frontend's exact origin in `FRONTEND_ORIGINS`.

The app has no login. Use a trusted network or an authenticated HTTPS proxy. Keep one Uvicorn worker because the counting queue is process-local.

## Development

Python dependencies are in `requirements.txt` and `requirements-dev.txt`. Run backend checks with `python -m pytest -q`; frontend and browser checks are under `tests/`.

See [VERIFICATION.md](VERIFICATION.md) for validation notes and [PERFORMANCE.md](PERFORMANCE.md) for benchmark history.
