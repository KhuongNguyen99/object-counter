# Countwell

Countwell counts objects in photos on your computer or private network. New counts use [CountGD++](https://github.com/niki-amini-naieni/CountGDPlusPlus); [FastSAM-s](https://docs.ultralytics.com/models/fast-sam/) helps suggest objects to select. Counting runs locally after the backend image is built, with no cloud inference or API key.

## Layout and requests

- `frontend/public/` contains the browser app. `frontend/Dockerfile` serves it from a small Nginx container on port 8080; the frontend image has no machine learning weights or PyTorch.
- `app/` contains the FastAPI backend. The root backend Dockerfile bundles the counting dependencies and assets; only this container does inference and stores counts. The backend root (`/`) returns 404, and its API documentation is at `/docs`.
- By default the browser calls the frontend origin. Nginx sends `/api/` and `/reports/` requests to `BACKEND_URL=http://counter:8080` on the Compose network. `frontend/public/config.js` supplies `apiBaseUrl`; the frontend container writes it from `API_BASE_URL` at startup. An empty value uses the Nginx proxy. `config.js` is served with no-store and the other static files with no-cache.

## Run with Docker

Install Python 3.11 or newer and start Docker with Linux containers and Docker Compose 2.30 or newer. On Windows, use Docker Desktop with its WSL2 backend. On Linux, use Docker Engine and the Compose plugin; NVIDIA GPU use also requires NVIDIA Container Toolkit. From the repository root, start the entire app with one command:

```powershell
# Windows
py -3 run.py
```

```sh
# Linux
python3 run.py
```

The launcher prepares and checks FastSAM-s using Python's standard library, then builds and starts the frontend and backend containers. The backend build downloads and verifies CountGD++ and BERT assets. No host machine learning packages or SAM3 checkpoint are needed. The first build needs an internet connection and downloads several large assets; unchanged builds use Docker's cache. Changes that invalidate asset preparation can repeat the large downloads. The launcher detects NVIDIA access through Docker and uses the GPU when available, otherwise the CPU. Add `--cpu` to force CPU mode or `--gpu` to require a working CUDA container; `--gpu` fails if that check fails. Add `--no-build` for a quick restart with already built images, without checking or downloading model assets.

The script starts the services in the background, waits for them to become healthy, and prints the app URL. Open **http://localhost:8080** by default. On a phone on the same Wi-Fi, use `http://YOUR-COMPUTER-LAN-IP:8080`; your computer may need to allow inbound TCP port 8080 on its private network. The backend API is exposed only on **http://127.0.0.1:8082** by default; its interactive documentation is at [http://127.0.0.1:8082/docs](http://127.0.0.1:8082/docs). The Camera button uses the browser's photo picker or camera, depending on the device.

To stop both services while keeping saved counts:

```sh
docker compose stop
```

You can also use Compose directly after the launcher has prepared the model assets. The CPU command is `docker compose up --build -d`; for an NVIDIA GPU with Docker GPU support enabled, use:

```sh
docker compose -f compose.yaml -f compose.gpu.yaml up --build -d
```

The GPU override applies only to the `counter` backend and uses CUDA 12.8 PyTorch. Keep both Compose files in later GPU commands. Run the regular CPU command with `--build` to switch back. Both use the unchanged `counter-data` volume. Sessions, uploaded photos, and annotated results survive restarts and `docker compose down`; **`docker compose down -v` deletes the volume**. Use `docker compose logs -f counter` or `docker compose logs -f frontend` to inspect errors. Copy `.env.example` to `.env` to change ports, binding addresses, and other settings described below.

### Private Tailscale access

On the configured host, open **https://artint-creare.tail11f4ee.ts.net:8081/** from a device on the same tailnet. This Tailscale Serve endpoint still forwards to the frontend at `127.0.0.1:8080`. The computer, Docker, and Tailscale must be running. On another host, check `tailscale serve status`, choose an unused HTTPS port, then configure:

```sh
tailscale serve --bg --https=8081 http://127.0.0.1:8080
```

Use the URL printed by that command. The Serve configuration survives Tailscale restarts; `tailscale serve --https=8081 off` removes this endpoint. See the [Tailscale Serve documentation](https://tailscale.com/docs/reference/tailscale-cli/serve).

### Separate frontend and backend

The frontend can be built or deployed independently. For a separate container using Nginx to proxy browser requests, set `BACKEND_URL` to a backend address reachable **from inside the frontend container** (origin only, with no `/api` path):

```sh
docker build -t countwell-frontend:local ./frontend
docker run --rm -p 8080:8080 -e BACKEND_URL=http://reachable-backend:8082 countwell-frontend:local
```

Alternatively, build the same frontend image with the browser calling a separate backend directly:

```sh
docker run --rm -p 8080:8080 -e API_BASE_URL=https://backend.example countwell-frontend:local
```

`API_BASE_URL` may include a path prefix; the frontend adds `/api/...` or `/reports/...` after it. The backend must be reachable from users' browsers and configured with `FRONTEND_ORIGINS` listing the exact frontend origin, for example `http://localhost:8080`. Multiple origins are comma-separated. Cross-origin requests have no credentials; the default allowed-origin list is empty. For plain static hosting, serve `frontend/public/` and set `globalThis.COUNTWELL_CONFIG = { apiBaseUrl: "https://backend.example" };` in its `config.js`. A separate backend can be bound through `API_BIND_ADDRESS` and `API_PORT` on a trusted network.

## Count objects

1. Add up to 20 photos. For each photo, choose **exactly 10 different examples** of the object you want counted. To get suggestions, choose **Mark areas**, draw one or more areas, then choose **Find objects**. Check the suggested selections; select or draw any missing examples. Nothing is sent for counting until you start it.
2. Check the outlines and selected object thumbnails. Click a thumbnail to focus its outline or use its **×** to deselect it. You can draw a box around an object, adjust a box, zoom, and undo. **Done** closes the editor; click a photo thumbnail to reopen it. Only selected examples guide that photo's count. The default **Match strength** is 0.80; a higher value keeps fewer, closer matches.
3. Choose **Start counting**. Inspect the numbered result for each photo, its count, the batch subtotal, and the running total. Use **Review photo** to add or remove counted objects when needed; saving corrections updates the totals and downloadable results.
4. Add another batch if needed. Each new photo needs its own 10 selections. Choose **Finish count** when done, then **New count** to start over. **Download results** exports the photo and batch breakdown as CSV.

**Quick count** sits beside **Start counting** and **Count this batch**. For a few separate, evenly spaced objects, mark one or more areas on every photo and choose Quick count. It uses the same automatic box finder and counts all retained boxes in those areas, without requiring ten examples. Overlapping areas have repeated boxes merged. The usual size filter still applies. Find objects and Quick count keep all eligible suggestions returned by the finder; the former 32-box cutoff is removed. Results use the same review, corrections, totals, saved history, and download flow. Match strength applies to regular counting.

Photos that overlap may count the same physical object twice. Exact repeat uploads within a count are rejected. A zero count does not prove the object is absent; inspect the result image. Overlapping, tiny, partly hidden, or unfamiliar objects can be missed or counted twice. Clear photos and varied, correctly boxed examples help, but every result still needs review. Failed batches do not add to the total and can be retried.

Uploads accept JPEG, PNG, and WebP, up to 20 MB and 30 megapixels per photo. The browser can also convert HEIC if it can decode it; otherwise export JPEG. Your current count restores on page reload. Download its results before starting a new count if you need a separate copy.

## Configuration and access

Copy `.env.example` to `.env` to configure the frontend `PORT` and `BIND_ADDRESS`, backend `API_PORT` and `API_BIND_ADDRESS`, `DEVICE` and `OMP_NUM_THREADS`, or the routing settings `BACKEND_URL`, `API_BASE_URL`, and `FRONTEND_ORIGINS`. Defaults are frontend port 8080 on all interfaces, backend port 8082 on localhost, automatic CPU/GPU selection, and four CPU threads. Set `BIND_ADDRESS=127.0.0.1` for computer-only frontend access.

The app has no login. Session links act as access keys, so use a trusted LAN or private tailnet. A reachable backend API also needs access control at the network or proxy layer; CORS is not authentication. For public access, place both services behind an authenticated HTTPS reverse proxy. Run one Uvicorn worker because its counting queue is process-local.

Existing saved counts from earlier model experiments can still be viewed, reviewed, and exported. New batches for those retired models are blocked. [PERFORMANCE.md](PERFORMANCE.md) and [VERIFICATION.md](VERIFICATION.md) record earlier benchmarks and limitations; those comparisons do not describe current model choices or establish unattended counting accuracy.

## Source, licenses, and local development

CountGD++ uses its official checkpoint and source pinned to commit `8489f836d75eb407ae778c56cff1f72cc6f8baae`. The backend image retains its source and licenses at `/opt/countgd`. FastSAM-s uses the official Ultralytics asset, verified against its pinned size and SHA-256 digest. See the [CountGD++ repository](https://github.com/niki-amini-naieni/CountGDPlusPlus) and [Ultralytics FastSAM documentation](https://docs.ultralytics.com/models/fast-sam/) for upstream source and license terms. Both assets are bundled into the backend for offline use after build.

For local CPU development, install Python 3.11, the CPU PyTorch wheels (`torch==2.9.1`, `torchvision==0.24.1`), and `requirements-dev.txt`. Prepare the FastSAM-s asset with `python scripts/download_models.py --model-dir models --models FastSAM-s`, prepare CountGD++ source with `python scripts/prepare_countgd.py --source-dir .vendor/countgd`, and set `COUNTGD_SOURCE_DIR` to the absolute `.vendor/countgd` path. Starting both services with Docker Compose is the easiest way to get the proxy settings right. To run them directly, set `FRONTEND_ORIGINS=http://localhost:8080` for the backend, set `frontend/public/config.js` to `apiBaseUrl: "http://localhost:8082"`, and start two terminals:

```sh
python -m uvicorn app.main:app --host 127.0.0.1 --port 8082 --workers 1
python -m http.server 8080 --directory frontend/public
```

Run `python -m pytest -q` for backend checks. The direct development setup calls the backend across origins; the Docker setup uses the frontend proxy.
