# A5000 frontend count after a verified warm hold

Tested through Chrome at the app's Tailscale frontend on October 22, 2026 (America/Chicago). A 3360 × 2156 parts photo with ten prepared reference boxes and 0.80 confidence returned **364 objects**.

**Photo upload to result: 8.3236 seconds. Photo upload to annotated image displayed: 10.1361 seconds.** These are one-trial frontend timings; the photo and references were prepared before the timer started.

## Ready state

One unmeasured CountGD++ request warmed the NVIDIA RTX A5000 model. The worker then remained running with no queued or active jobs for **30.815 seconds**, confirmed by 34 consecutive health samples. The hold used a monotonic timer because the UTC clock stepped during the test. The timed frontend request ran on the **same worker** as the warm-up and reported `cold_start=false`. Warm-up and the ready hold are outside the frontend timings below.

## Time breakdown

The upload timer begins immediately before the frontend sends the photo batch.

| Stage | Seconds |
| --- | ---: |
| Photo upload and acceptance | 0.1770 |
| RunPod queue/dispatch delay | 0.1850 |
| RunPod request execution | 5.0450 |
| Other app, transfer, and polling time (calculated remainder) | 2.9166 |
| **Photo upload to result received** | **8.3236** |
| Additional annotated image loading, decoding, and display | 1.8125 |
| **Photo upload to annotated image displayed** | **10.1361** |

The full **Start counting** click to result took 8.3656 seconds, including session creation before upload. The worker handler took 3.995 seconds, including 3.739 seconds for counting; these are nested within RunPod execution. The remainder is calculated by subtraction, not independently measured, and browser and cloud stage boundaries can overlap.

For the same photo and references, the previous idle/cold-worker trial took 28.02 seconds from upload to result and 29.91 seconds from upload to display. Its RunPod queue/startup delay was 18.851 seconds. Both trials are single observations and may vary with network and provider load.

## Cost and cleanup

The controller recorded a quoted A5000 rate of **$0.69 per GPU-hour**. At that rate, the verified 30.815-second ready hold alone represents an **estimated $0.00591** of GPU time (about 0.59 US cents). This is a hold-only estimate, not a billed amount or a total test cost. Warm-up/preparation, timed execution, scale-down, and storage are outside that estimate; this record does not establish their combined bill.

The endpoint completed one warm-up and one timed request with no failures or retries. After the test, minimum workers was restored to **0**, maximum workers to **1**, and idle timeout to **5 seconds**. RunPod reported **zero running workers** and no queued or active jobs. The original frontend script was restored and its SHA-256 verified.

Raw evidence is in the local Codex test workspace, outside this NAS repository: `C:\Users\phape\Documents\Codex\2026-10-21\ri\test-results\a5000-warm-frontend-test\`. See `report.md`, `report.json`, `controller-report.json`, `browser-events.json`, `warmup-job.json`, `timed-job.json`, and `result.png`. These local artifacts are listed as file names rather than repository links.
