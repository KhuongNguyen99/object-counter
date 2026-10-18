# CountGD++ GPU execution record

The `gpu-execution-v2` CountGD++ path reduces warm, cached full-image inference time on the measured fitting photo from **4.122 s to 3.669 s** (six alternating paired trials, median; **11.0% less time, 1.12× speedup**). Every timed run produced the same 367 merged detections and the same merged-result hash. This is a performance and output-equivalence result for the tested fixture, not a claim that 367 is the correct physical count or that other photos will gain the same speedup.

## What changed

The cached CountGD++ inference path skips decoder intermediate heads and loss metadata that counting never consumes, uses host-known spatial shapes and prompt token lists to avoid repeated CUDA scalar synchronization, and transfers the selected output in a packed operation. The original official `forward` remains available. Model weights, float32 execution, input resolution, 800-pixel crops, 24-tile layout, overlap, merge, confidence threshold, reference conditioning, and model/batch cache behavior are unchanged. The YOLOE runtime is unchanged.

The baseline is local commit `ee2b687` (tag `gpu-baseline`), with image `yoloe-counter:gpu-baseline` at image ID `d2f3617426981b203e47bc0c1a2aa7db0a10133d35bea1b4a09146e85f6210d4`. The candidate is the `gpu-execution-v2` source patch and image tag `yoloe-counter:gpu-execution-v2`. Both use the official CountGD++ source commit `8489f836d75eb407ae778c56cff1f72cc6f8baae` and the same checkpoint SHA-256 recorded in the benchmark metadata.

## Measurement and exactness

The fixed fitting target is the original 4000 × 2252 photo in `test-results/fitting-debug/target-original.jpg`. The reference is `reference-original.jpg`, with box `(1039, 1775, 1344, 2170)`, label `Bolt with red cap`, and confidence `0.5`. It ran on an NVIDIA GeForce RTX 4090 Laptop GPU with 16 GiB VRAM, PyTorch 2.9.1+cu128, and four CPU compute threads. The model was loaded and warmed before timing. Each timed full-image call includes reference preparation, image loading, 24 crop passes, merging, and annotation; it excludes model loading and warmup. CUDA is synchronized at the timing boundaries.

| Warm paired trials | Baseline | Candidate |
| --- | ---: | ---: |
| Median of six full-image runs | 4121.8 ms | 3669.0 ms |
| Merged detections in every run | 367 | 367 |

The order alternated baseline then candidate, followed by candidate then baseline. All 12 timed results had the same merged-result SHA-256, `b0374fa5d16b31cf262c5896b378b1dd3a9b30361a335bd0f06c10e10e6d42f2`. The paired timings are in the ignored local artifact `test-results/gpu-execution/paired-timings.json`. A separate one-tile profiler capture reduced `cudaStreamSynchronize` calls from 239 to 70; profiler call counts explain the mechanism but are not full-image timings.

The full capture in `test-results/gpu-execution/comparison-fitting/comparison.json` passed an exact comparison across all **900 queries × 24 tiles = 21,600 queries**. Raw boxes, logits, sigmoid scores, tokens, selection masks, selected boxes and scores, and merged detection records matched byte for byte; the annotated JPEG also matched exactly in the separate file comparison. The fixed comparison's five-run medians were 3886.1 ms and 3662.4 ms, a separate measurement from the alternating pairs. Peak allocated CUDA memory was 1695.4 MiB for each image.

The white-ring comparison in `test-results/gpu-execution/comparison-ring.json` also passed exact raw-output and merged-result checks, with 101 merged detections on each side. An independent visual review counted 75 physical rings, so this fixture retains the same accuracy errors on both paths. The optimization does not fix missed, duplicate, or false detections.

An SDPA decoder-attention experiment had a 3.47 s median over five sequential runs, but changed raw logits (maximum finite difference `0.07303166389465332`) and failed the exact-output comparison in `comparison-sdpa.json`. It was rejected and is not in the candidate. The host suite passed 125 tests with two skips, the browser handler suite passed 11, the offline GPU suite passed 127, and all five actual YOLOE CUDA checkpoints were exercised. These checks cover the tested configuration; laptop clock and thermal variation affect timing, and one fitting photo does not establish a general speedup.

## Reproduce

The benchmark tool is [`scripts/compare_gpu_execution.py`](scripts/compare_gpu_execution.py). Its `run` command requires an empty output directory and saves `summary.json`, a compressed `queries.npz`, and annotations. `compare` reads two run directories and writes `comparison.json` and `comparison.md`; it exits nonzero if the metadata or any required output differs. The default fixture, box, label, confidence, and model directory are the values above. The images, model assets, and `test-results/` artifacts are intentionally excluded from Git; supply the original photos and local images before repeating the run.

On a Docker host with NVIDIA GPU support, set `fixture_root` to the absolute, writable host path for `test-results` that the Docker daemon can mount. Set `benchmark_script` to the absolute host path for the current comparison script. The baseline image predates that script, so bind-mount the same script into both images. Run each image in its own container, with no other GPU workload during the measurement:

```sh
fixture_root=/absolute/path/to/test-results
benchmark_script=/absolute/path/to/scripts/compare_gpu_execution.py
docker run --rm --gpus all --mount "type=bind,src=${fixture_root},dst=/app/test-results" --mount "type=bind,src=${benchmark_script},dst=/app/scripts/compare_gpu_execution.py,readonly" yoloe-counter:gpu-baseline python scripts/compare_gpu_execution.py run --name baseline --device 0 --repeats 5 --output-dir /app/test-results/gpu-execution/repro-baseline
docker run --rm --gpus all --mount "type=bind,src=${fixture_root},dst=/app/test-results" --mount "type=bind,src=${benchmark_script},dst=/app/scripts/compare_gpu_execution.py,readonly" yoloe-counter:gpu-execution-v2 python scripts/compare_gpu_execution.py run --name candidate --device 0 --repeats 5 --output-dir /app/test-results/gpu-execution/repro-candidate
docker run --rm --mount "type=bind,src=${fixture_root},dst=/app/test-results" --mount "type=bind,src=${benchmark_script},dst=/app/scripts/compare_gpu_execution.py,readonly" yoloe-counter:gpu-execution-v2 python scripts/compare_gpu_execution.py compare --baseline /app/test-results/gpu-execution/repro-baseline --candidate /app/test-results/gpu-execution/repro-candidate --output-dir /app/test-results/gpu-execution/repro-comparison
```

For the paired timing method, start these in separate interactive terminals, using the same `fixture_root` and `benchmark_script` values and fresh output directories:

```sh
docker run --rm -i --gpus all --mount "type=bind,src=${fixture_root},dst=/app/test-results" --mount "type=bind,src=${benchmark_script},dst=/app/scripts/compare_gpu_execution.py,readonly" yoloe-counter:gpu-baseline python scripts/compare_gpu_execution.py run --name baseline --device 0 --worker --repeats 6 --output-dir /app/test-results/gpu-execution/paired-baseline
docker run --rm -i --gpus all --mount "type=bind,src=${fixture_root},dst=/app/test-results" --mount "type=bind,src=${benchmark_script},dst=/app/scripts/compare_gpu_execution.py,readonly" yoloe-counter:gpu-execution-v2 python scripts/compare_gpu_execution.py run --name candidate --device 0 --worker --repeats 6 --output-dir /app/test-results/gpu-execution/paired-candidate
```

Wait for `READY baseline` and `READY candidate`. Send trial index `1` to baseline then candidate, index `2` to candidate then baseline, and continue alternating through index `6`; send `quit` to each when finished. Each worker performs a full-image warmup first, holds its model in memory, and writes `trial-01.json` through `trial-06.json`. Use the six `wall_ms` values per image to calculate medians and check every `merged` record. The standalone `compare` command above checks full query arrays from ordinary `run` captures; worker trials record merged results and timings.

For a service rollback after the inference queue is idle, restore the preserved baseline image under the Compose tag and recreate the service:

```sh
docker image tag yoloe-counter:gpu-baseline yoloe-counter:cuda128
docker compose -f compose.yaml -f compose.gpu.yaml up --no-build -d
```

This does not remove the `counter-data` volume. Avoid `docker compose down -v` when preserving sessions.
