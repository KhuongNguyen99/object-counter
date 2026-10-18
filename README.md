# Reference Object Counter

A phone-friendly counting app with a FastAPI backend and a lightweight browser UI, shipped together in one Docker image. **CountGD++ (experimental)** is the default, alongside all five **YOLOE-26** visual-prompt models (Nano, Small, Medium, Large, Extra Large). All weights, the CountGD++ source, and its BERT encoder are included. No cloud inference, API key, or runtime model download is needed.

## Run

```sh
docker compose up --build -d
```

Open **http://localhost:8080** on your computer. On a phone connected to the same Wi-Fi, open `http://YOUR-COMPUTER-LAN-IP:8080`. Allow inbound TCP port 8080 on your computer's private network if needed. The camera button uses the phone's native photo picker/camera; browser and phone behavior may vary. Gallery upload also works on desktop.

The first build downloads Python packages and any missing model weights. The weights are baked into the image. Counting sessions, uploaded photos, and annotated results are stored in the `counter-data` Docker volume and survive container restarts. `docker compose down` keeps them; `docker compose down -v` removes them.

### Tailscale access

On this computer, open **https://artint-creare.tail11f4ee.ts.net:8081/** from a device connected to the same Tailscale network. The computer, Tailscale, and Docker app must be running. Access follows your tailnet's permissions.

Tailscale Serve on the Docker host provides HTTPS and forwards to the app's local port. Port 8081 keeps the host's existing Serve endpoints available. To configure this on another host, first inspect `tailscale serve status` and choose an unused HTTPS port, then run:

```sh
tailscale serve --bg --https=8081 http://127.0.0.1:8080
```

Use the URL printed by the command. The background configuration survives Tailscale restarts. To remove just this endpoint, run `tailscale serve --https=8081 off`. See the [Tailscale Serve documentation](https://tailscale.com/docs/reference/tailscale-cli/serve).

### NVIDIA GPU

With Docker GPU support enabled (NVIDIA Container Toolkit on Linux, or Docker Desktop with WSL2 and supported NVIDIA drivers on Windows):

```sh
docker compose -f compose.yaml -f compose.gpu.yaml up --build -d
```

This builds `yoloe-counter:cuda128` with CUDA 12.8 PyTorch and requests the GPU for both CountGD++ and YOLOE. Keep both Compose files in subsequent GPU deployment commands. The regular Compose command builds the smaller `yoloe-counter:local` CPU image; run it again with `--build` to switch back. Both use the same session volume. The selected GPU deployment survives container and Docker restarts.

CountGD++ GPU optimization measurements, exact-output checks, reproduction commands, and rollback steps are recorded in [PERFORMANCE.md](PERFORMANCE.md).

For later rebuilds from this NAS workspace, reuse the model files already in the GPU image to avoid transferring them again. The build still verifies every checkpoint, and the resulting image includes all assets:

```sh
docker build --build-context model-assets=docker-image://yoloe-counter:cuda128 --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 -t yoloe-counter:cuda128 .
docker compose -f compose.yaml -f compose.gpu.yaml up --no-build -d
```

## Count objects

1. Describe the object and choose a model. For CountGD++, the Object field is also a text prompt: use a description such as **white plastic ring**. CountGD++ is the default trial model. YOLOE uses the visual references; its Nano variant favors speed, while larger sizes use more memory and compute.
2. Add one or more reference photos. The Object description guides automatic boxes around examples of the **same kind of object**. Review the numbered boxes, remove unwanted ones, or draw a box manually. If you add photos before entering a description, enter it and choose **Auto-detect**. References guide detection; they do not train the model or add to the total.
3. Take a photo or select one or more counting photos, then start counting.
4. Review the numbered annotations and count on **each image**, the batch subtotal, and the session's running total. Detection details include the confidence of each numbered object. Zero detections are flagged **Review needed**; they do not prove that no objects are present.
5. Add the next photos to accumulate another batch using the same references and model. Choose **Finish counting** when done, then start a new count for another object. A CSV export preserves the image and batch breakdown.

Photograph new objects in each batch. The app adds detections across photos; it does not track physical objects between photos. Overlapping views can count the same object again. Exact repeated images within a session are rejected to help prevent accidental duplicate uploads. Failed batches do not change the total and can be retried. Your current session restores on reload; export its CSV before starting a new count.

Use clear reference examples at a similar scale and viewpoint to your targets. A lower confidence threshold can recover missed objects but also add false or duplicate detections. Larger models do not guarantee accurate counts of unfamiliar parts. CountGD++ substantially improved the supplied white-ring examples, but still missed some overlapping rings and added extra detections. Its results remain marked **Review needed**. These two photos do not establish accuracy on new scenes. See [VERIFICATION.md](VERIFICATION.md) for the investigation. Always inspect the annotations before accepting a count.

## Models and downloads

The [official YOLOE documentation](https://docs.ultralytics.com/models/yoloe/) describes the visual-prompt models used here:

| Size | Included checkpoint |
| --- | --- |
| Nano | `yoloe-26n-seg.pt` |
| Small | `yoloe-26s-seg.pt` |
| Medium | `yoloe-26m-seg.pt` |
| Large | `yoloe-26l-seg.pt` |
| Extra Large | `yoloe-26x-seg.pt` |

[CountGD++](https://github.com/niki-amini-naieni/CountGDPlusPlus) uses the official `countgd_plusplus.pth` checkpoint and BERT assets under `models/countgd/`. Its source is pinned to commit `8489f836d75eb407ae778c56cff1f72cc6f8baae`; the original source and licenses are preserved at `/opt/countgd` in the image. A compatibility patch selects the repository's existing PyTorch attention implementation when its compiled extension is absent. All model assets are checked against pinned SHA-256 digests before loading.

The downloader verifies official checkpoint checksums. To populate `models/` before building:

```sh
python scripts/download_models.py --model-dir models
```

YOLOE visual references are encoded and combined into one target class. CountGD++ uses the object description and boxed references; examples from multiple reference photos are arranged into one contact sheet. Extra references can change both misses and false detections, so inspect the annotations. The runtime loads one selected model at a time and serializes inference to avoid GPU memory contention and prompt state mixing between sessions. CountGD++ uses its bundled offline BERT encoder. It prepares the reference image, reference features, and text features once per counting batch, then reuses them across its tiles and images. Each tile receives fresh text state for image/text fusion; a new batch prepares new conditioning. This cache works on CPU and GPU without changing resolution, overlap, confidence, or model precision.

Automatic reference boxes use the bundled CountGD++ detector in text-only mode, independently of the model selected for counting. Its proposed boxes use original-photo coordinates and become ordinary editable references for either family. Nested part proposals are suppressed in favor of a complete enclosing object; edge-clipped and near-full-frame proposals are excluded. This is a suggestion step, and the description may match the wrong object or miss an unfamiliar one. The minimum-confidence slider controls counting, while reference suggestions use 0.23. Empty results leave the photo available for manual drawing. Temporary proposal uploads are removed when processing finishes; accepted reference images are saved only when the count is created. Reference detection shares the single worker and CountGD++ cache with counting, and requires no additional model download or service.

Large photos are automatically counted in overlapping crops whenever the selected model would otherwise reduce the source resolution. YOLOE uses crops up to its configured input size (640×640 by default); CountGD++ uses up to 800×800. Crops use the original pixels with at least 20% overlap. Neighboring detections are translated into the original photo's coordinates and reconciled before the final count; tile subtotals are not added directly. The result shows one full-resolution annotated photo, a tile count, and merge information inside Details. Larger photos take longer. This handles repeated views caused by tiling; the model can still miss objects or produce incorrect detections. Overlapping photos in separate uploads remain additive.

## Configuration

Copy `.env.example` to `.env` to change `PORT`, `BIND_ADDRESS`, `DEVICE`, or `OMP_NUM_THREADS`. Defaults are port 8080, all network interfaces, automatic device selection, and four CPU compute threads. Use `BIND_ADDRESS=127.0.0.1` for computer-only access.

This app is intended for your computer, a trusted LAN, or your private Tailscale network. Session links act as access keys; there is no app login. Keep tailnet access limited to trusted users. Use an authenticated HTTPS reverse proxy for public access.

JPEG, PNG, and WebP images are supported, up to 20 MB and 30 megapixels per image, 8 references, 32 reference rectangles, and 20 counting images per batch. The browser also converts HEIC when it can decode it; otherwise export the photo as JPEG. Originals are normalized for EXIF orientation before inference. Interrupted jobs become failed on restart and can be retried; finished counts remain saved.

YOLOE uses a 640-pixel input size and permits up to 3,000 detections per crop. Advanced deployments can pass `IMGSZ` (32–4096) or `MAX_DETECTIONS` (1–30000) as container environment variables for YOLOE; larger values need more memory and time. CountGD++ uses its official 800-pixel short-side / 1333-pixel maximum resize on each crop, with 900 detection queries per crop. Warnings appear if any crop reaches its model's limit; the merged photo can contain more objects than a single crop's limit. Photos requiring more than 256 crops are rejected with a clear error; use a less extreme image aspect ratio.

Useful commands:

```sh
docker compose logs -f counter
docker compose ps
docker compose stop
```

## Local development and checks

Python 3.11 is used by the image. For local CPU development:

```sh
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
python scripts/download_models.py --model-dir models
python scripts/prepare_countgd.py --source-dir .vendor/countgd
# Set COUNTGD_SOURCE_DIR to the absolute .vendor/countgd path before starting Python locally.
python -m pytest -q
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --workers 1
```

Use exactly **one Uvicorn worker**: the inference queue and active model are process-local. API tests inject a deterministic inference engine; set `RUN_MODEL_TESTS=1` to also exercise all five real YOLOE checkpoints. These checkpoint tests default to CPU; set `MODEL_TEST_DEVICE=0` to verify each size on CUDA. `RUN_COUNTGD_TESTS=1` enables the local CountGD++ comparison when the benchmark fixtures are present and follows `DEVICE`. Run `node --test tests/test_frontend.mjs` for the browser-handler regressions. Real model, Docker, and desktop/mobile browser verification is recorded in [VERIFICATION.md](VERIFICATION.md).
