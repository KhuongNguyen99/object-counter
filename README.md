# Countwell

Countwell counts objects in photos on your computer or private network. New counts use [CountGD++](https://github.com/niki-amini-naieni/CountGDPlusPlus); [FastSAM-s](https://docs.ultralytics.com/models/fast-sam/) helps suggest objects to select. Both run locally after the Docker image is built. Counting does not require a cloud service or API key.

## Run with Docker

This checkout needs `models/FastSAM-s.pt` and `models/countgd/` in the build context. The build verifies their contents and prepares the pinned CountGD++ source and offline text encoder. If you have a fresh checkout without those assets, use Python 3.11 to prepare them before building:

```sh
python -m pip install -r requirements.txt
python scripts/download_models.py --model-dir models --models FastSAM-s
python scripts/prepare_countgd.py --source-dir .vendor/countgd
```

The first download is large. The Docker build also downloads any missing CountGD++ assets and verifies their checksums. No SAM3 checkpoint is needed.

For CPU counting:

```sh
docker compose up --build -d
```

Open **http://localhost:8080**. On a phone on the same Wi-Fi, use `http://YOUR-COMPUTER-LAN-IP:8080`; your computer may need to allow inbound TCP port 8080 on its private network. The Camera button uses the browser's photo picker or camera, depending on the device.

For an NVIDIA GPU with Docker GPU support enabled (NVIDIA Container Toolkit on Linux, or Docker Desktop with WSL2 and supported drivers on Windows):

```sh
docker compose -f compose.yaml -f compose.gpu.yaml up --build -d
```

Keep both Compose files in later GPU commands. The GPU build uses CUDA 12.8 PyTorch. Run the regular CPU command with `--build` to switch back. Both deployments use the same `counter-data` volume.

Sessions, uploaded photos, and annotated results survive container restarts and `docker compose down`. **`docker compose down -v` deletes the data volume.** Use `docker compose logs -f counter` to inspect startup or counting errors.

### Private Tailscale access

On the configured host, open **https://artint-creare.tail11f4ee.ts.net:8081/** from a device on the same tailnet. The computer, Docker, and Tailscale must be running. Access follows the tailnet's permissions. On another host, check `tailscale serve status`, choose an unused HTTPS port, then configure Tailscale Serve:

```sh
tailscale serve --bg --https=8081 http://127.0.0.1:8080
```

Use the URL printed by that command. The Serve configuration survives Tailscale restarts; `tailscale serve --https=8081 off` removes this endpoint. See the [Tailscale Serve documentation](https://tailscale.com/docs/reference/tailscale-cli/serve).

## Count objects

1. Add up to 20 photos. For each photo, choose **exactly 10 different examples** of the object you want counted. To get suggestions, choose **Mark areas**, draw one or more areas, then choose **Find objects**. Check the suggested selections; select or draw any missing examples. Nothing is sent for counting until you start it.
2. Check the outlines and selected object thumbnails. Click a thumbnail to focus its outline or use its **×** to deselect it. You can draw a box around an object, adjust a box, zoom, and undo. **Done** closes the editor; click a photo thumbnail to reopen it. Only selected examples guide that photo's count. The default **Match strength** is 0.80; a higher value keeps fewer, closer matches.
3. Choose **Start counting**. Inspect the numbered result for each photo, its count, the batch subtotal, and the running total. Use **Review photo** to add or remove counted objects when needed; saving corrections updates the totals and downloadable results.
4. Add another batch of photos if needed. Each new photo needs its own 10 selections. Choose **Finish count** when done, then **New count** to start over. **Download results** exports the photo and batch breakdown as CSV.

**Quick count** sits beside **Start counting** and **Count this batch**. For a few separate, evenly spaced objects, mark one or more areas on every photo and choose Quick count. It uses the same automatic box finder and counts all retained boxes in those areas, without requiring ten examples. Overlapping areas have repeated boxes merged. The usual size filter still applies. Find objects and Quick count keep all eligible suggestions returned by the finder; the former 32-box cutoff is removed. Results use the same review, corrections, totals, saved history and download flow. Match strength applies to regular counting.

Photos that overlap may count the same physical object twice. Exact repeat uploads within a count are rejected. A zero count does not prove the object is absent; inspect the result image. Overlapping, tiny, partly hidden, or unfamiliar objects can be missed or counted twice. A clear photo and varied, correctly boxed examples help, but every result still needs review. Failed batches do not add to the total and can be retried.

Uploads accept JPEG, PNG, and WebP, up to 20 MB and 30 megapixels per photo. The browser can also convert HEIC if it can decode it; otherwise export JPEG. Your current count restores on page reload. Download its results before starting a new count if you need a separate copy.

## Configuration and access

Copy `.env.example` to `.env` to change `PORT`, `BIND_ADDRESS`, `DEVICE`, or `OMP_NUM_THREADS`. Defaults are port 8080, all network interfaces, automatic CPU/GPU selection, and four CPU threads. Set `BIND_ADDRESS=127.0.0.1` for access from this computer only.

The app has no login. Session links act as access keys, so use a trusted LAN or private tailnet. For public access, place it behind an authenticated HTTPS reverse proxy. Run one Uvicorn worker: its counting queue is process-local.

Existing saved counts from earlier model experiments can still be viewed, reviewed, and exported. New batches for those retired models are blocked. [PERFORMANCE.md](PERFORMANCE.md) and [VERIFICATION.md](VERIFICATION.md) record earlier benchmarks and limitations; those comparisons do not describe model choices in the current app or establish unattended counting accuracy.

## Source and licenses

CountGD++ uses its official checkpoint and source pinned to commit `8489f836d75eb407ae778c56cff1f72cc6f8baae`. The image retains its source and licenses at `/opt/countgd`. FastSAM-s uses the official Ultralytics asset, verified against its pinned size and SHA-256 digest. See the [CountGD++ repository](https://github.com/niki-amini-naieni/CountGDPlusPlus) and [Ultralytics FastSAM documentation](https://docs.ultralytics.com/models/fast-sam/) for upstream source and license terms. Both assets are bundled for offline use after build.

For local CPU development, install Python 3.11, the CPU PyTorch wheels (`torch==2.9.1`, `torchvision==0.24.1`), and `requirements-dev.txt`. Prepare the two assets as above, set `COUNTGD_SOURCE_DIR` to the absolute `.vendor/countgd` path, then run `python -m pytest -q` or `python -m uvicorn app.main:app --host 127.0.0.1 --port 8080 --workers 1`.
