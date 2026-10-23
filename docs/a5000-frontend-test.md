# A5000 frontend counting test

Recorded October 22, 2026 (America/Chicago), through Chrome at the app's Tailscale address.

One CountGD++ request returned **364 objects** from a 3360 x 2156 photo, using ten prepared reference boxes and 0.80 confidence. The A5000 worker started from a scaled-down state with the container image already cached.

**Result received by the frontend: 28.17 seconds. Annotated photo displayed: 30.06 seconds. Estimated total cost: $0.006481, approximately 0.65 US cents.**

## Time breakdown

The timer starts when the user clicks **Start counting**.

| Stage | Duration |
| --- | ---: |
| Create session and upload photo | 0.411 s |
| Cloud waiting and worker startup | 18.851 s |
| Cloud request processing | 6.368 s |
| Other app, transfer and polling time (calculated remainder) | 2.536 s |
| **Completed result received by frontend** | **28.166 s** |
| Additional annotated image loading, decoding and display | 1.893 s |
| **Annotated photo displayed** | **30.059 s** |

Counting itself took 5.048 seconds within the 6.368-second processing period. Worker preloading took 8.274 seconds within the startup delay. These nested durations are already included in the totals.

The 2.536-second remainder is calculated by subtraction. Browser and cloud metrics use different stage boundaries, so this trace cannot split it precisely between network, local processing and polling. Photo preparation took 0.499 seconds before the click and is excluded; the reference boxes were also ready before timing.

## Estimated cost breakdown

The quoted A5000 rate was **$0.69/hour**. Container-start logs and scale-down observations span 31.179 seconds, rounded up to **32 seconds** for the compute estimate.

| Component | Basis | Estimated cost (USD) |
| --- | --- | ---: |
| Request processing | 6.368 seconds | $0.001221 |
| Worker startup, idle/shutdown overhead and rounding | 25.632 seconds | $0.004913 |
| Container storage | 30 GB, one five-minute block at $0.10/GB/month | $0.000347 |
| **Total for this request** | **Includes startup and shutdown** | **$0.006481** |

**Billing was pending when this test was recorded. This is an estimate, not a confirmed charge or a formal upper bound.** Container logs and health checks approximate the billing boundaries. The estimate covers this request's running interval and storage; earlier setup attempts and requests are excluded.

## Verification

The frontend received the completed result and displayed its annotated image. The saved result contained 364 detections. Exactly one additional cloud job completed, with no failures or retries. Afterward, the endpoint reported zero running workers and no queued or active jobs.

Temporary timing hooks were removed after the test, and the original frontend file was restored and verified. Single-trial timing can vary with photo size, network conditions, caching and GPU availability.

See the [A5000/L4 comparison](runpod-benchmark.md) for the earlier GPU test. Billing references: [RunPod pricing](https://docs.runpod.io/serverless/pricing) and [worker lifecycle](https://docs.runpod.io/serverless/workers/overview).
