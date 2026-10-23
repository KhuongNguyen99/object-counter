# Warm A5000 test with results drawn in the browser

Tested on 2026-10-23 through the normal Chrome frontend with one warm-up, a **30.703-second** ready hold, and one timed count. The worker reused the same NVIDIA RTX A5000, reported `cold_start=false`, and ran build `67320726a3c1d43ed86d17360ba11bca5b666c3e`. The 3360 × 2156 input had ten reference boxes at 0.80 confidence. The normalized image was 2,641,157 bytes, SHA-256 `5961f264294e4646e724351fe9602f4c0d7d4794acebb4fa32fe0a7016a97806`, matching the prior warm cloud test. The count and all 364 detection coordinates and scores matched exactly.

**Photo upload to completed result: 6.364 seconds. Photo upload to displayed annotated result: 6.450 seconds.** Chrome `performance.now()` is the primary timing clock; photo selection and reference setup happened earlier.

## Timing

| Measurement | Seconds |
| --- | ---: |
| Click to photo upload | 0.082 |
| Photo upload to batch acceptance | 0.363 |
| Batch acceptance to completed result | 6.001 |
| RunPod queue or dispatch | 0.191 |
| RunPod execution | 5.060 |
| Worker handler within execution | 3.932 |
| CountGD within handler | 3.662 |
| **Photo upload to completed result received** | **6.364** |
| Result received to count DOM | 0.005 |
| Result received to SVG image displayed | 0.086 |
| **Photo upload to SVG image displayed** | **6.450** |
| Click to SVG image displayed | 6.531 |

The worker and CountGD spans are nested inside cloud execution. The single Linux transport request lasted 6.126 seconds; its boundary differs from the Chrome timeline. The timed transport completed in one `/runsync` POST with no status polls.

The current UI overlays annotations in SVG on a cached photo. Its display marker observed the embedded SVG image load, decoded the cached blob, verified a connected and visible SVG with 365 rectangles and 365 text elements (364 detections plus the badge), then waited two animation frames. The result has no annotation JPEG URL. The earlier warm frontend trial waited for a generated annotated JPEG, so the display measurements cover different presentation workflows. The photo-upload-to-completed-result bounds are comparable, subject to normal single-trial variation.

| Frontend measurement | Original warm A5000 | Optimized warm A5000 | This metadata UI A5000 | Prior warm local |
| --- | ---: | ---: | ---: | ---: |
| Photo upload to result | 8.324 s | 6.792 s | 6.364 s | 3.510 s |
| Photo upload to displayed result | 10.136 s | 8.197 s | 6.450 s | 3.754 s |

Compared with the optimized warm A5000 trial, this trial's upload-to-result interval was 6.31% shorter and upload-to-display interval was 21.32% shorter. The cloud execution interval was 5.060 seconds here versus 4.898 seconds previously. These are single observations across different app workflows; they do not isolate a causal saving from any one code change, establish GPU acceleration, or rank GPUs.

## Estimated endpoint cost

The live A5000 setup quote was **$0.69/hour**. Basis: observed container-start log to first verified zero-running health response. The observed active interval was 121.157 seconds, rounded to 122 seconds for this estimate. It includes the successful endpoint's warm-up, ready hold, timed execution, startup, other idle time, and shutdown allowance. This is an estimate, not a provider receipt or formal upper bound.

| Component | Seconds | Estimated USD |
| --- | ---: | ---: |
| Warm-up execution | 6.589 | $0.001263 |
| Ready hold | 30.703 | $0.005885 |
| Timed execution | 5.060 | $0.000970 |
| Remaining startup, other idle, shutdown and rounding | 79.648 | $0.015266 |
| **Compute estimate** | **122** | **$0.023383** |
| 30 GB storage allowance, 2 modeled five-minute blocks | — | $0.000694 |
| **Estimated total** | | **$0.024078** |

| Successful endpoint estimate | Prior optimized warm A5000 | This metadata UI warm A5000 |
| --- | ---: | ---: |
| Rounded compute interval | 96 s | 122 s |
| Compute | $0.018400 | $0.023383 |
| Storage allowance | $0.000694 | $0.000694 |
| **Full successful endpoint estimate** | **$0.019094** | **$0.024078** |
| Timed execution within compute | $0.000939 | $0.000970 |

The full estimate is higher because this test's observed active interval was longer: it includes more idle and operator handoff time, plus the interval before zero running workers was observed. The controller saw 17.969 seconds from ready release to upload on approximately comparable same-host UTC timestamps, and 46.250 seconds from completion observation to zero-running observation on its monotonic clock. That latter span can include provider scale-down or health-reporting lag; it is **not** a measurement of GPU-active time. The controller's UTC and monotonic total elapsed times agreed within 0.001290 seconds, but any provider clock offset remains unknown. The timed-execution estimate itself changed from $0.000938783 to $0.000969833. The prior warm local run incurred no RunPod charge; local hardware and power are excluded from this comparison.

Provider billing is pending; no provider records yet; no finalized receipt is available. An empty provider response is not evidence of zero cost. The storage allowance covers endpoint creation through deletion, 386.611 seconds, with five-minute blocks on the same basis as the prior warm report. Timestamp boundaries and reporting lag make this approximate. The quote and allowance follow [RunPod's pricing documentation](https://docs.runpod.io/serverless/pricing). This estimate covers the successful endpoint only, not other account activity.

## Verification and artifacts

The normal app session has one completed batch with 364 detections and no annotation JPEG URL. The saved backend access log contains no `/files/` image fetch. The endpoint completed exactly two jobs without failures or retries. Its minimum workers were restored to zero, zero running workers were observed, the test endpoint was deleted and returned HTTP 404, and both temporary frontend and backend diagnostics were removed. The configured production endpoint, its minimum-zero setting, and zero queued and in-progress jobs were rechecked.

Local source provenance records uncommitted metadata UI changes; the worker image was digest pinned as `ghcr.io/khuongnguyen99/object-counter-worker@sha256:aa014b294edca296acf87340191155bad70d5eed0d3dc6b4d046350337118395`. Raw artifacts and `report.json` are in `C:\Users\phape\Documents\Codex\2026-10-21\ri\test-results\a5000-metadata-warm-frontend-test`. Browser markers are in `browser-events.json`, sanitized cloud timing in `timed-job.json`, worker warm-up in `warmup-job.json`, and cleanup in `controller-report.json` and `final-verification.json`.
