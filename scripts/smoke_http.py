"""Exercise a running container with a real image, accumulation, and annotated results."""

import argparse
import io
import json
from pathlib import Path
import time

from PIL import Image, ImageOps
import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--image", type=Path, required=True, help="810x1080 Ultralytics bus.jpg sample")
    parser.add_argument("--model", default="yoloe-26n-seg")
    parser.add_argument("--output", type=Path, default=Path("test-results"))
    args = parser.parse_args()
    base = args.url.rstrip("/")
    args.output.mkdir(parents=True, exist_ok=True)
    http = requests.Session()

    def call(method, path, **kwargs):
        response = http.request(method, base + path, timeout=30, **kwargs)
        response.raise_for_status()
        return response

    catalog = call("GET", "/api/models").json()
    assert len([m for m in catalog["models"] if m["family"] == "YOLOE-26"]) == 5
    assert all(m["available"] for m in catalog["models"])
    original = args.image.read_bytes()
    session = call("POST", "/api/sessions", data={
        "label": "Verification - people", "model": args.model, "confidence": ".20",
        "reference_boxes": json.dumps([[[48, 398, 244, 902]], [[221, 405, 345, 858]]]),
    }, files=[("references", ("reference-a.jpg", original, "image/jpeg")),
              ("references", ("reference-b.jpg", original, "image/jpeg"))]).json()
    path = f"/api/sessions/{session['id']}"

    with Image.open(io.BytesIO(original)) as image:
        mirrored = io.BytesIO()
        ImageOps.mirror(image).save(mirrored, format="JPEG", quality=94)
        smaller = io.BytesIO()
        image.resize((648, 864)).save(smaller, format="JPEG", quality=94)

    counts = []
    for batch_files in ([('bus.jpg', original), ('mirror.jpg', mirrored.getvalue())],
                        [('resized.jpg', smaller.getvalue())]):
        queued = call("POST", path + "/batches", files=[
            ("images", (name, content, "image/jpeg")) for name, content in batch_files
        ]).json()
        deadline = time.monotonic() + 240
        while True:
            session = call("GET", path).json()
            batch = next(b for b in session["batches"] if b["id"] == queued["id"])
            if batch["status"] == "failed":
                raise AssertionError(batch["error"])
            if batch["status"] == "complete":
                break
            assert time.monotonic() < deadline, "Counting timed out"
            time.sleep(.5)
        assert len(batch["results"]) == len(batch_files)
        assert batch["total_count"] == sum(r["count"] for r in batch["results"])
        counts.append(batch["total_count"])
        assert session["total_count"] == sum(counts)
        for result in batch["results"]:
            assert result["count"] == len(result["detections"])
            annotated = call("GET", result["annotated_url"]).content
            with Image.open(io.BytesIO(annotated)) as image:
                assert image.size == (result["width"], result["height"])
            (args.output / f"annotated-{result['filename']}").write_bytes(annotated)

    duplicate = http.post(base + path + "/batches", files={
        "images": ("repeated.jpg", original, "image/jpeg")}, timeout=30)
    assert duplicate.status_code == 409
    csv = call("GET", path + "/export.csv").text
    assert len(csv.splitlines()) == 4
    session = call("POST", path + "/finish").json()
    assert session["status"] == "finished" and session["image_count"] == 3
    assert session["total_count"] == sum(counts)
    (args.output / "http-session.json").write_text(json.dumps(session, indent=2), encoding="utf-8")
    print(json.dumps({"session_id": session["id"], "model": args.model, "batch_counts": counts,
                      "total": session["total_count"], "images": 3, "status": "passed"}))


if __name__ == "__main__":
    main()
