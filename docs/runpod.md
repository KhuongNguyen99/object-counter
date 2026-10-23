# RunPod Serverless

The browser and API run locally; CountGD++ counting and FastSAM reference-box suggestions run on a RunPod queue endpoint. Counting uses reference boxes drawn on each target image. Original image bytes are sent to the worker; annotations are drawn locally after the response.

## Launch

Set these values in the existing, ignored `.env` file:

```dotenv
INFERENCE_BACKEND=runpod
RUNPOD_API_KEY=<your RunPod API key>
RUNPOD_ENDPOINT_ID=<your queue endpoint ID>
RUNPOD_TIMEOUT_SECONDS=600
RUNPOD_BENCHMARK_BUDGET_USD=5.00
```

```powershell
python run.py --runpod
```

To use another private settings file, add `--env-file .env.runpod`. Both `.env` and `.env.*` are excluded from Git and Docker build contexts.

Use an endpoint running this repository's worker image. The endpoint must have minimum workers **0** and maximum workers **1** for the setup described here. The API key stays on the local backend; do not put it in frontend settings, commit it, or include it in an image. Remote mode supports CountGD++ per-image counting and FastSAM suggestions.

## Publish the worker

Run the manually dispatched **Publish RunPod worker** workflow, defined in `.github/workflows/runpod-worker.yml`, against the desired Git revision:

```powershell
gh workflow run runpod-worker.yml --ref <branch-or-tag>
```

The workflow builds the CUDA backend, bundles model assets, adds `Dockerfile.runpod`, and pushes:

```text
ghcr.io/khuongnguyen99/object-counter-worker:<full-Git-SHA>
```

Before publishing, the workflow verifies model hashes and file access as the nonroot worker user. For code-only patches, its optional `base_image` input can reuse an existing worker image pinned by digest; changes to dependencies, model assets, or upstream model source need a fresh base build.

It uses the workflow's `GITHUB_TOKEN` with `packages: write`; a separate registry password is unnecessary. Ensure the GHCR package is public so RunPod can pull it without a registry credential. Get the immutable image digest from the Docker push log or:

```powershell
docker buildx imagetools inspect ghcr.io/khuongnguyen99/object-counter-worker:<full-Git-SHA>
```

Use `ghcr.io/khuongnguyen99/object-counter-worker@sha256:<digest>` for the benchmark. Each response must report the expected build revision and identical model hashes. A mutable image tag is rejected by the benchmark.

## Compare A5000 and L4

See the [completed A5000/L4 comparison](runpod-benchmark.md) for measured timings, costs, and startup behavior.

Create a fixture manifest with a `cases` array. Each case supplies a unique `id`, `action` (`count` or `suggest`), an `image` path relative to the manifest, and an optional `label`. Count cases also supply `boxes` in `[x1, y1, x2, y2]` pixel coordinates and `confidence: 0.80`. Use the actual selected boxes on that same image. Add both count and suggest cases for each representative original image.

```powershell
python scripts/runpod_benchmark.py --image ghcr.io/khuongnguyen99/object-counter-worker@sha256:<digest> --expected-build-revision <full-Git-SHA> --fixture test-results/runpod-benchmark/fixture.json --output-dir test-results/runpod-benchmark/comparison --budget-usd 5
```

The defaults select **RTX A5000** and **NVIDIA L4**, with three warm repeats after each case's first call. The tool queries current GPU pools and prices, excludes other cards, verifies the deployed settings and actual worker GPU, and verifies that warm repeats use the same worker. Only the endpoint's first request is a full cold start; later cases have their first-call timings reported separately.

Each endpoint has one GPU, minimum zero workers, maximum one worker, a 30-second idle timeout, and a 300-second execution timeout. New endpoints require CUDA 12.8 or newer through `gpu.minCudaVersion`, matching the worker's PyTorch CUDA build; the setting is verified before jobs run. Trials are limited to 900 seconds per GPU. Before creating resources, the tool reserves the quoted full trial lifetime plus cleanup time and a 25% margin against the $5 budget. These are local safeguards, **not a provider-enforced account spending cap**. The default comparison runs GPUs in sequence. For independent trials, pass `--gpu "RTX A5000"` or `--gpu "NVIDIA L4"` once and use a different output directory for each run; the caller must keep their combined spending within the total budget. Fresh image downloads can take several minutes. Use `--job-seconds 1800 --trial-seconds 1800` when allowing a longer first startup.

Reports and the ownership manifest are saved under the output directory:

- `report.json` and `report.md`: hardware, image/build/model identity, cold and warm timings, and costs.
- `samples.csv` and `costs.csv`: timings and cost allocations for analysis.
- `resources.json`: only this run's endpoint creation intents and IDs, retained for cleanup recovery.

Both endpoints are deleted on completion by default. `--keep-winner` retains at most one endpoint with verified zero active workers and minimum zero workers, selected by the lowest sum of warm median request times. It does not select by finalized cost. Copy its reported ID into `RUNPOD_ENDPOINT_ID` when ready to use it.

To create a separate app endpoint after the comparison, run `python scripts/runpod_deploy.py create --image <image-at-digest> --gpu "RTX A5000" --manifest test-results/runpod-production/resources.json`. This uses zero minimum workers, one maximum worker, a five-second idle timeout, and standard FlashBoot. The comparison disables FlashBoot to measure full cold starts.

## Billing and cleanup

Provider billing records include GPU, CPU, and container-disk amounts. They can lag; empty records mean **pending**, not free. The report also gives a conservative estimate for the entire endpoint lifetime, including startup and idle. Execution time alone is never presented as total cost.

Refresh the receipt without creating resources:

```powershell
python scripts/runpod_benchmark.py --refresh-billing --output-dir test-results/runpod-benchmark/comparison
```

If an endpoint was retained, refresh its benchmark receipts **before sending production traffic**: later jobs on that endpoint would enter its billing totals. The billing API exposes no finalization flag, so observed receipts remain labeled accordingly.

Normal failures trigger cleanup. After an interruption, network failure, or forcibly killed process, verify cleanup using the saved manifest:

```powershell
python scripts/runpod_deploy.py cleanup --manifest test-results/runpod-benchmark/comparison/resources.json
```

This also deletes any retained winner. Keep the manifest until cleanup completes. The tool verifies endpoint name and image before deleting; changed resources require review. An uncertain create is reconciled by its unique saved name rather than submitted again.

## Image size limits

The app sends up to 10 MB of serialized JSON through `/run` and larger requests, up to 20 MB, through `/runsync`. Base64 adds roughly one third to image size, so an original image near **15 MB** can exceed the limit once JSON and boxes are included. Oversized images are rejected without recompression; larger originals require an object-storage transfer implementation. The benchmark currently uses the stricter 10 MB serialized limit, roughly **7.5 MB** of original image bytes.

Official references: [endpoint creation](https://docs.runpod.io/api-reference-v2/serverless/create-a-serverless-endpoint), [GPU catalog](https://docs.runpod.io/api-reference-v2/catalog/list-gpu-types), [billing history](https://docs.runpod.io/api-reference-v2/billing/get-serverless-billing-history), [billing phases](https://docs.runpod.io/serverless/pricing), and [request operations](https://docs.runpod.io/serverless/endpoints/operation-reference).
