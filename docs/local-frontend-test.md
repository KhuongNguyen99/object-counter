# Local frontend timing and warm RunPod comparison

On October 23, 2026 (America/Chicago), the local **RTX 4090 Laptop GPU** returned **364 objects**. Sending the photo to receiving the result took **3.5097 seconds**; the annotated image was visible after **3.7537 seconds**.

The previous [warm A5000 test](a5000-warm-frontend-test.md) is recorded separately. Both tests used the same 3360 × 2156 parts photo, ten reference boxes, and 0.80 confidence. The server-normalized image bytes matched exactly: 2,641,157 bytes, SHA-256 `5961f264294e4646e724351fe9602f4c0d7d4794acebb4fa32fe0a7016a97806`. Both counted 364 objects using 20 tiles and removed 334 overlapping tile detections.

## Frontend time breakdown

These durations use Chrome's performance timer. Photo preparation and example selection happened before timing.

| Stage | Local PC | Warm RunPod A5000 |
| --- | ---: | ---: |
| Send photo to upload acceptance received | 0.1489 s | 0.1770 s |
| Upload acceptance to result received | 3.3608 s | 8.1466 s |
| **Send photo to result received** | **3.5097 s** | **8.3236 s** |
| Additional image loading, decoding, and display | 0.2440 s | 1.8125 s |
| **Send photo to annotated image displayed** | **3.7537 s** | **10.1361 s** |
| Start counting click to result received | 3.5511 s | 8.3656 s |

The result boundary is the completed batch response parsed by the frontend. Display waits for the annotated image to load and decode, followed by two animation frames. Upload acceptance is the browser's receipt of the HTTP 202 response, before the separate `batch_accepted` log marker. Counting runs in the background and can overlap the upload response. The normal 1.6-second result polling interval is included.

## Local processing detail

These are nested measurements from the Linux container's own performance timer. They explain the work inside the request; they are **not additional frontend time**.

| Measurement | Reported duration |
| --- | ---: |
| Queue submission to batch start | 0.000398 s |
| Batch start to result saved in the database | 3.470667 s |
| Inference engine call, within the batch | 3.400312 s |
| CountGD++ counting stage, within the engine | 3.31088 s |
| Prepare reference conditioning, within counting | 0.365140 s |
| Twenty tile prediction calls combined, within counting | 2.831619 s |
| Cached model lookup, within the engine | 0.000089 s |
| After inference: final image save and database completion | 0.065006 s |

The counting stage includes target decoding, reference conditioning, tile prediction, merging, and the first annotated image save. It excludes model loading and exemplar preparation. The corresponding A5000 counting-stage reading was 3.73927 seconds.

Browser totals are the primary end-to-end measurements. Backend durations use a separate Linux monotonic clock. Those duration measurements do not fully reconcile, and Linux UTC also stepped during the run; backend timings are retained as independent diagnostics. They must not be added to the frontend stages or subtracted to claim an exact network/polling cost. This also limits precise hardware-only speed comparisons.

## Preparation and test conditions

One unmeasured count warmed the local model and returned 364. Its container-reported batch duration was 44.195 seconds. The initialized service then stayed ready for **30.266 seconds**, verified with 30 successful health samples using the Windows monotonic timer. The timed request confirmed that the model was already cached in the same process.

The local test used Chrome at `http://127.0.0.1:8083/`, Nginx, FastAPI, and Docker GPU access. PyTorch was 2.9.1 with CUDA 12.8; the NVIDIA driver was 596.49. The CountGD++ checkpoint SHA-256 was `b89395d79711c9b9f13f3ef80cd2e528386fb1d70453f99443be4eccab33592f`. Both setups used the PyTorch attention fallback.

The cloud trial used the same PC's frontend through its Tailscale HTTPS URL and called RunPod. The local trial used loopback HTTP and local inference. The elapsed-time difference includes network, cloud dispatch, app polling, and image delivery as well as processing. Each setup has one timed warm trial; laptop power and temperature, network conditions, and load can change the result.

The prepared fixture button seeded the same ten boxes before timing. The timed count followed the normal Start counting, multipart upload, result polling, and image rendering flow. Temporary timing logs were added only to isolated app copies; production source and the normal app connection were unchanged.

## Cost and cleanup

The local count added **$0 in RunPod compute fees** and sent **no new RunPod inference jobs**. Electricity and hardware costs were not measured. The cloud warm-hold estimate is recorded in the [A5000 report](a5000-warm-frontend-test.md); it is not an actual billing receipt.

Both temporary local test containers were stopped after saving the result. RunPod remained at minimum workers **0**, with **zero running workers** and no active or queued jobs; its completed-job total stayed at six.

Raw evidence is saved outside the repository at `C:\Users\phape\Documents\Codex\2026-10-21\ri\test-results\local-warm-frontend-test\`: `report.json`, `browser-events.json`, `backend-events.jsonl`, `session-result.json`, `ready.json`, `environment.json`, `input-comparison.json`, `cloud-cleanup.json`, and `result.png`.
