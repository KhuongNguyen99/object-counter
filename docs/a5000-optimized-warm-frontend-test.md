# Optimized warm A5000 frontend test

Tested on October 23, 2026 (America/Chicago), through the same Tailscale HTTPS frontend as the previous warm A5000 trial.

One Chrome frontend request returned **364 objects** from a 3360 × 2156 photo with ten reference boxes at 0.80 confidence. The timed count used the same digest-pinned A5000 worker as a successful warm-up, after a **30.478-second** ready hold. Its worker reported `cold_start=false` and build revision `67320726a3c1d43ed86d17360ba11bca5b666c3e`. The server-normalized photo matched the prior warm cloud input: 2,641,157 bytes, SHA-256 `5961f264294e4646e724351fe9602f4c0d7d4794acebb4fa32fe0a7016a97806`.

**Photo upload to completed result: 6.792 seconds. Photo upload to annotated image displayed: 8.197 seconds.** These are Chrome `performance.now()` durations; photo selection and reference setup happened earlier.

## Timing

| Measurement | Seconds |
| --- | ---: |
| Photo upload to batch acceptance | 0.263 |
| RunPod queue/dispatch | 0.275 |
| RunPod execution | 4.898 |
| Worker handler within execution | 3.910 |
| CountGD within handler | 3.646 |
| **Photo upload to result received** | **6.792** |
| Result received to image displayed | 1.405 |
| **Photo upload to image displayed** | **8.197** |

The worker and CountGD spans are nested within cloud execution. The saved WSL transport trace has 1 request(s), totaling 5.966 seconds of request durations; those Linux spans have different boundaries and are not added to the Chrome timeline.

This version uses `/runsync` for ordinary photos, checks pending browser counts every 500 ms, and skips the discarded annotation on the model worker. The timed cloud transport completed in **one `/runsync` POST with no status-poll requests**. The final annotated image was still drawn locally. The saved comparison also confirms exactly identical detection boxes, scores, and count to the previous warm A5000 result for this photo.

| Frontend measurement | Prior warm A5000 | This warm A5000 | Prior warm local |
| --- | ---: | ---: | ---: |
| Photo upload to result | 8.324 s | 6.792 s | 3.510 s |
| Photo upload to displayed image | 10.136 s | 8.197 s | 3.754 s |

These are single cases across different paths. They do not establish the saving from any individual code switch or rank the GPUs.

## Estimated endpoint cost

The A5000 quote in setup was **$0.69/hour**. Basis: observed container-start log to first verified zero-running health response. The observed interval was 95.517 seconds, rounded up to 96 seconds for this estimate. This is an active-window estimate for the warm-up, ready hold and timed request together, **not a billing receipt or a formal upper bound**.

| Component | Seconds | Estimated USD |
| --- | ---: | ---: |
| Warm-up execution | 6.164 | $0.001181 |
| Ready hold | 30.478 | $0.005842 |
| Timed execution | 4.898 | $0.000939 |
| Remaining startup, other idle, shutdown and rounding | 54.460 | $0.010438 |
| **Compute estimate** | **96** | **$0.018400** |
| 30 GB storage allowance, 2 modeled five-minute blocks | — | $0.000694 |
| **Estimated total** | | **$0.019094** |

Provider billing is pending; no provider records yet; no finalized receipt is available. Two jobs belong to this new test endpoint, so any provider records are endpoint-only observations for the warm-up and timed trial together, not a finalized per-job charge.

The image-cache loading interval before the container-start log is outside the compute estimate. This avoids pricing all endpoint preparation time as GPU-running time. The observed shutdown boundary can include reporting lag, so the estimate remains approximate.

The storage allowance uses the endpoint's full creation-to-deletion interval of 372.145 seconds, rounded to 2 modeled five-minute blocks. This includes preparation and cache loading; the blocks are an allowance, not observed billing entries or a formal cap.

The billing calculation follows [RunPod's pricing documentation](https://docs.runpod.io/serverless/pricing): compute is rounded to seconds, and container storage is priced in five-minute blocks. The GPU quote itself was read from the live catalog and saved in `setup.json`.

An earlier preparation attempt on endpoint `6a6rf7in629nba` stopped after a monitoring connection error, before a timed frontend request. All 18 saved health samples showed zero running workers; a later connection gap prevents treating those samples as a complete billing history. That endpoint was deleted and a subsequent management read returned HTTP 404. Its provider receipt still has no records. **The estimate above covers the successful endpoint only; the aborted attempt's actual charge remains pending and is not assumed to be zero.** Its separate records are retained in `preparation-attempt-1/` and `preparation-attempt-1-verification.json`.

## Verification and artifacts

The normal app session contains one completed batch with 364 detections at the expected dimensions. The endpoint recorded exactly two completed jobs with no failures or retries. Minimum workers was restored to zero, zero running workers and no queued jobs were verified, and the owned endpoint was deleted. The source image is hash-verified; the worker image was `ghcr.io/khuongnguyen99/object-counter-worker@sha256:aa014b294edca296acf87340191155bad70d5eed0d3dc6b4d046350337118395`.

Raw artifacts and report.json are saved outside the repository at `C:\Users\phape\Documents\Codex\2026-10-21\ri\test-results\a5000-optimized-warm-frontend-test\`. Browser timing is in `browser-events.json`, the cloud jobs in `warmup-job.json` and `timed-job.json`, cleanup in `controller-report.json`, and the separate transport trace in `transport.json`. The original frontend and backend diagnostic copies were restored after the test. The app uses the updated production worker image with minimum workers zero.
