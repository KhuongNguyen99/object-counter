# RunPod GPU comparison

A5000 is configured for the app. It processed CountGD++ requests faster on both test photos at the same quoted GPU rate as L4. L4 was faster at suggesting reference boxes. This is a small test of one worker per GPU, with three warm repeats per case.

Tested on October 22, 2026 (America/Chicago). Each GPU completed 16 requests: counting and box suggestions for two photos, with four calls per case. Both used identical original image bytes, selected boxes, empty labels, and 0.80 counting confidence. The photos were 3360 Ã— 2156 pixels with ten references and 4000 Ã— 2252 pixels with one reference.

## Speed

Warm medians in seconds. Processing measures time inside the worker handler. Full request time also includes upload, queueing, polling, and download.

| Task | A5000 processing | L4 processing | A5000 full request | L4 full request |
| --- | ---: | ---: | ---: | ---: |
| Count parts | 3.93 | 5.70 | 8.69 | 9.06 |
| Count fittings | 4.98 | 7.05 | 9.95 | 11.00 |
| Suggest boxes, parts | 0.289 | 0.210 | 4.66 | 2.31 |
| Suggest boxes, fittings | 0.289 | 0.232 | 5.09 | 5.30 |

Averaging the two counting medians, A5000 used **30% less worker processing time** and **7% less full request time**. Those measurements support choosing A5000 for counting; they do not establish a fleet-wide performance guarantee. Network variation was substantial for these short requests.

Both GPUs returned 364 parts, 365 fittings, 268 parts proposals, and 257 fittings proposals. Each quantity stayed the same across all repeats. Coordinates and scores can differ between GPUs. These checks measure consistency, not accuracy against a manually labeled ground truth.

The completed trials' first requests took about **65 seconds** on both GPUs, including worker startup. Earlier fresh-image attempts took about four to five minutes; one download exceeded the 15-minute client wait limit. Both timed trials disabled FlashBoot. The app endpoint enables standard FlashBoot to help subsequent starts. Cold startup depends on image caching and available capacity.

## Cost

RunPod quoted **$0.69 per GPU-hour** for both cards during setup. The following are conservative estimates using the entire endpoint lifetime, including image download and queue time that may be unbilled. They are separate from provider receipts and include startup and idle time.

| Scope | Lifecycle estimate |
| --- | ---: |
| A5000 completed comparison, 16 requests | $0.046 |
| L4 completed comparison, 16 requests | $0.046 |
| All seven comparison attempts, including failures and retries | $1.293 |
| App setup and three verification requests, through verified scale-down | $0.066 |
| **Entire setup and comparison** | **$1.36** |

Each completed comparison therefore has an estimated batch cost of about **$0.0029 per request**, with its startup and idle overhead divided across 16 requests. This is not the price of a single isolated request.

At the final billing check, available provider receipts totaled **$0.004399** for one failed attempt; records for the other seven endpoints, including the app check, were still pending. **The final billed total is not available yet.** Empty receipts are not treated as zero. The $1.36 lifecycle estimate remains below the authorized $5 budget; budget reservations are not charges. Later use of the app is outside this setup comparison.

## App verification

The local frontend and lightweight API now use the A5000 endpoint. The local API contains neither PyTorch nor model weights. Through the frontend proxy, the live check passed reference suggestions within a marked crop, standard counting (364 objects), Quick count within a marked area (7 objects), annotated image downloads, and CSV exports.

All seven comparison endpoints were deleted. The retained app endpoint has minimum zero workers, maximum one worker, a five-second idle timeout, and standard FlashBoot. After verification, RunPod reported zero running workers and no queued or active jobs. Parked workers may remain listed as Idle or Initializing; those states are scaled down according to [RunPod's worker lifecycle](https://github.com/runpod/docs/blob/main/serverless/workers/overview.mdx).

Private settings are in `.env.runpod`, excluded from Git and Docker build contexts. Start the app from the repository root with:

```sh
python run.py --runpod --env-file .env.runpod
```

## Reproduce

See [setup and benchmark commands](runpod.md). The comparison used the same worker image and model assets on both cards, with PyTorch's fallback attention implementation. [Measurements as CSV](runpod-benchmark.csv) contain the per-case medians and cost estimates. Raw reports and ownership manifests stay in the ignored `test-results/runpod-benchmark/` folders.

- Worker image: `ghcr.io/khuongnguyen99/object-counter-worker@sha256:4ff749999363e72cf1a710ad42c3dcbcceee6bf50ddd2cb0200ee9d39b4b8d97`
- Worker build: `cbff702d7e0d4869cbb8ceec2b59602b96901a54`
- Fixture SHA-256: `bf23bee7f6859968052fa523efbf0dd8c2269f4819ed8c3377e92e4ee3ac7823`
- CountGD++ weights SHA-256: `b89395d79711c9b9f13f3ef80cd2e528386fb1d70453f99443be4eccab33592f`
- FastSAM-s weights SHA-256: `c9f78716a81c7aff0d608ccc73e1b82ab3aaad86005049f6a92106a0be6d0844`

Billing references: [pricing and billable phases](https://docs.runpod.io/serverless/pricing), [billing history API](https://docs.runpod.io/api-reference-v2/billing/get-serverless-billing-history).
