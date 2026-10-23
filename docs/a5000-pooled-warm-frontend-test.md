# Pooled HTTPS warm A5000 frontend trial

The first endpoint was aborted after its browser warmup timed out while queued. It completed no job and exposed no running worker; its separate allowance is below. A fresh endpoint restricted to CA-MTL-1 then ran the successful trial. Two separate attempt-2 Chrome pages counted the same original 3360 × 2156 parts photo with ten reference boxes and 0.80 confidence through the normal UI. The first page warmed the frontend, backend HTTPS client, and worker. After a **30.156-second** ready hold, the second page made the timed count. Both jobs returned 364 detections from worker `k1u776lthmscrv` on an NVIDIA RTX A5000. Active worker metadata reported actual placement in **CA-MTL-1**. The warmup reported `cold_start=true`; the timed job reported `cold_start=false`.

**Timed photo upload to completed result: 5.451 seconds. Timed photo upload to displayed annotated result: 5.612 seconds.** The first page's corresponding intervals were 6.911 and 7.034 seconds. Chrome `performance.now()` is the primary frontend clock within each page; preparation happened before either count click.

## Timed path

| Measurement | Seconds |
| --- | ---: |
| Click to photo upload | 0.067 |
| Photo upload to batch acceptance | 0.151 |
| Batch acceptance to result | 5.300 |
| RunPod queue or dispatch | 0.066 |
| RunPod execution | 4.549 |
| Worker handler within execution | 4.132 |
| CountGD within handler | 3.801 |
| **Photo upload to completed result** | **5.451** |
| Result to count DOM | 0.004 |
| Result to displayed SVG | 0.161 |
| **Photo upload to displayed SVG** | **5.612** |

The result marker observed the count in the DOM. The display marker observed the local photo blob decoding and the embedded SVG image load, a connected visible SVG with 365 rectangles and 365 text elements, then two animation frames. The app returned no generated annotation JPEG URL, and the backend access log has no `/files/` image fetch.

## HTTPS and worker SDK timing

The same backend `RunPodClient` produced run ordinals 1 and 2. Its `/runsync` POST used a **new** connection for warmup and a **reused** connection for the timed job after the hold. Connect time changed from 33.92 to 0.00 ms; the respective HTTPS request totals were 6533.33 and 5244.38 ms. This states observed connection reuse; a held process alone cannot prove the remote peer kept the socket open.

| HTTP phase | Warmup ms | Timed ms |
| --- | ---: | ---: |
| DNS, TCP, TLS on fresh connection | 33.92 | 0.00 |
| Request upload | 73.45 | 24.33 |
| Wait for response headers | 6314.14 | 5129.45 |
| Response body read | 93.93 | 90.17 |
| Full HTTPS request | 6533.33 | 5244.38 |

| Worker SDK phase after inference | Warmup ms | Timed ms |
| --- | ---: | ---: |
| Handler to result handoff | 2.73 | 3.64 |
| Full SDK result handoff | 70.71 | 55.30 |
| Before result POST, including JSON encoding and SDK setup | 1.33 | 1.70 |
| Result POST upload and acknowledgement, including any retries | 69.33 | 53.50 |
| After result POST | 0.05 | 0.10 |

Worker SDK timing comes from job-hash-matched container log records. Its spans, the HTTPS request, RunPod queue and execution, and Chrome markers have different boundaries and cannot be added as one waterfall. The source photo is 2,266,233 bytes (SHA-256 `cf6ee072c10fff38d95ba6cff4bddd4c248619f3ff7c5faef17a77d53f0734f9`); both server-normalized inputs were 2,641,157 bytes (SHA-256 `5961f264294e4646e724351fe9602f4c0d7d4794acebb4fa32fe0a7016a97806`), with the same ten boxes and confidence. The timed detections and scores matched the previous benchmark exactly.

| Frontend measurement | Prior metadata A5000 | This pooled A5000 | Prior warm local |
| --- | ---: | ---: | ---: |
| Photo upload to result | 6.364 s | 5.451 s | 3.510 s |
| Photo upload to displayed result | 6.450 s | 5.612 s | 3.754 s |

Relative to the metadata A5000 trial, the pooled trial's upload-to-result interval changed by -14.34% and upload-to-display changed by -12.98% (positive means longer). The previous A5000 warmup used a separate client, leaving its timed backend HTTPS client fresh; this trial warms the same client and worker through the frontend with the same normalized photo. The previous A5000 worker's data center was not recorded, so a closer-region effect cannot be isolated. The prior local RTX 4090 Laptop 16 GB trial used loopback, while this cloud trial used Tailscale HTTPS. These comparisons cover the whole path, not pure GPU speed or the effect of one change.

## Full successful endpoint estimate

The live A5000 quote was **$0.69/hour**. The successful endpoint's **primary conservative estimate** applies that quote to the 284.891-second controller window from the minimum-one request to the first verified zero-running response, rounded to 285 seconds. A container-start log was unavailable, so this window includes initialization and may exceed the actual running-worker interval. Its storage allowance covers the **entire 338.805-second endpoint lifetime**. The previous metadata trial started at an observed container-start log, so its primary cost is not directly comparable with this wider window.

| Component | Seconds | Estimated USD |
| --- | ---: | ---: |
| Warmup execution | 5.946 | $0.001140 |
| Ready hold | 30.156 | $0.005780 |
| Timed execution | 4.549 | $0.000872 |
| Remaining controller window, including initialization, idle, shutdown and rounding | 244.349 | $0.046834 |
| **Primary compute estimate** | **285** | **$0.054625** |
| 30 GB storage allowance, 2 modeled five-minute blocks | — | $0.000694 |
| **Primary successful endpoint estimate** | | **$0.055319** |

As a **qualified alternative**, the successful worker record had status `RUNNING` and `startedAt=2026-10-23T18:35:16.82Z`. From that metadata timestamp to the first zero-running response is 121.721494 seconds, rounded to 122 seconds. The same quote gives $0.023383 for compute plus $0.000694 for full-lifetime storage, or **$0.024078**. `startedAt` is a worker metadata proxy that may change across lifecycle states; provider documentation does not identify it as a billing start timestamp. The controller-envelope estimate above remains primary.

For the aborted first endpoint, the separate conservative allowance applies the A5000 quote to its entire 503.374-second lifetime even though no running worker was observed. It first waited in US-IL-1 with no observed GPU worker, then the same endpoint was restricted to CA-MTL-1 and one worker appeared in `INITIALIZING` state; no second job was submitted. Its hypothetical compute is $0.096600; storage allowance is $0.000694; total is **$0.097294**. Provider billing for that endpoint is pending. This allowance is not an observed charge or a claim that initialization was billed.

Adding that aborted allowance to the successful endpoint's primary controller-envelope estimate gives a **combined trial estimate of $0.152614**. Combining the qualified worker-metadata proxy with the same aborted allowance instead gives **$0.121372**. The aborted allowance hypothetically applies the quote to the full lifetime. The successful endpoint's own full-lifetime quote envelope is $0.065669, shown separately from its primary estimate.

The previous metadata trial's primary estimate was $0.024078, using an observed container-start-to-zero-running interval; its timed execution portion was $0.000970, versus $0.000872 here. Its cost boundary differs from the current primary controller window. The prior local trial incurred no RunPod charge; its hardware and power costs are excluded. For the successful endpoint, the provider returned no billing records yet, and billing is not finalized. A final provider refresh also found no billing records for the aborted endpoint. Empty responses are not evidence of zero cost. These are estimates, not billing receipts or formal upper bounds. The pricing basis follows [RunPod's pricing documentation](https://docs.runpod.io/serverless/pricing).

## Verification and artifacts

The successful endpoint completed exactly two jobs without failures or retries. Minimum workers returned to zero, zero running workers were observed, and it was deleted. The independent watchdog did not start because its helper file had been moved during the first-attempt archive; the controller's nested cleanup completed these actions. A final management API audit returned HTTP 404 for both the successful and aborted endpoints. Temporary frontend and backend diagnostics were removed; the saved production endpoint remained minimum-zero with an empty queue. The successful controller's UTC and monotonic elapsed times agreed within 0.001649 seconds. Its completion-to-zero-running interval includes provider scale-down or health-reporting lag and is not a measurement of GPU-active time.

The worker image was digest pinned as `ghcr.io/khuongnguyen99/object-counter-worker@sha256:8af667a071640dc4cb50cb57b2c895fab5451be9c8300d15e35dc5ccf8fba6c5` at build revision `1168bae62d135c41b4e37eec88b3c1c8059e4ac5`. The local client fix was committed separately at `f3e815629e7a88310233a8d26807e706dbedde30` and did not change the worker image. Successful raw Chrome markers, both sanitized cloud jobs and transport traces, hash-correlated worker timing logs, placement metadata, cleanup evidence, and `report.json` are in `C:\Users\phape\Documents\Codex\2026-10-21\ri\test-results\a5000-pooled-warm-frontend-test`. The first endpoint's evidence is in `attempt-1/`. This is one successful case, so it does not establish a causal saving or rank GPUs.
