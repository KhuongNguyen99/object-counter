"""Render saved hardware-counting results as one offline, self-contained HTML report.

No inference runs here. Usage:
  python scripts/render_reliability_report.py --input-root <experiment directory>
"""

from __future__ import annotations

import argparse
import base64
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path.home() / "AppData/Local/Temp/counter-reliability-20261021"
DEFAULT_OUTPUT = ROOT / "test-results/reliability-benchmark/report.html"
EXPECTED = ("baseline-tune", "strategy-output", "background-output", "mixed-tune",
            "heldout-core", "heldout-mixed", "heldout-core-per-object", "heldout-mixed-per-object",
            "negative-prompts", "additional-models", "sam3-threshold-0.5", "sam3-threshold-0.7",
            "sam3-threshold-0.9", "normalization-core", "normalization-mixed")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def study_design():
    names = ("manifest.json", "mixed-hardware-manifest.json")
    base = ROOT / "test-results/reliability-benchmark/data"
    targets = [target for name in names
               for target in json.loads((base / name).read_text(encoding="utf-8"))["targets"]]
    return {split: {"photos": len({t["source_sha256"] for t in targets if split == "all" or t["split"] == split}),
                    "queries": sum(split == "all" or t["split"] == split for t in targets)}
            for split in ("all", "tune", "heldout")}


def phase(source):
    top = source.split("/", 1)[0]
    if top.startswith("normalization-"):
        return "JPEG normalization stress"
    if top.startswith("sam3-threshold-"):
        return "SAM3 tune threshold sweep"
    if top.endswith("-per-object"):
        return "Exploratory heldout prototype"
    if top.startswith("heldout-"):
        return "Frozen heldout"
    return {"baseline-tune": "Frozen tune baseline", "mixed-tune": "Mixed tune baseline",
            "strategy-output": "Tune strategy screen", "background-output": "RGB reference context",
            "negative-prompts": "Negative prompt screen", "additional-models": "Additional model screen"}.get(top, top)


def compact_record(record, key, source, report_sha, manifest_sha):
    prediction = record.get("prediction") or {}
    metrics = record.get("metrics") or {}
    target = record.get("target") or {}
    settings = record.get("settings") or {}
    experiment = settings.get("experiment") or {}
    strategy = experiment.get("name") if isinstance(experiment, dict) else str(experiment)
    strategy = strategy or record.get("modality") or "baseline"
    if (record.get("model") == "yoloe-26x-seg"
            and settings.get("secondary_reference_strategy") == "per-object"):
        strategy += " / per-object prototypes"
    audit_fields = ("model", "threshold", "variant", "target_id", "category", "split",
                    "target_sha256", "points", "gt_boxes", "references", "label", "settings",
                    "status", "target", "prediction", "metrics", "error", "wall_ms",
                    "started_at", "finished_at")
    audit_record = {field: record[field] for field in audit_fields if field in record}
    return {
        "id": hashlib.sha256((source + "/" + key).encode()).hexdigest()[:16],
        "run": source.rsplit("/", 1)[0], "phase": phase(source),
        "source": source, "source_sha256": report_sha,
        "manifest_sha256": manifest_sha, "record_key": key,
        "model": record.get("model", "?"), "threshold": record.get("threshold"),
        "strategy": strategy,
        "variant": record.get("variant", "default"),
        "category": record.get("category", "?"), "split": record.get("split", "?"),
        "target_id": record.get("target_id", "?"), "target_sha256": target.get("sha256", record.get("target_sha256", "")),
        "truth": len(record.get("points") or []), "predicted": prediction.get("count"),
        "matched": metrics.get("matched"), "missed": metrics.get("missed"),
        "extra": metrics.get("extra"), "exact": metrics.get("exact_image"),
        "count_equal": prediction.get("count") == len(record.get("points") or []) if prediction else None,
        "elapsed_ms": prediction.get("elapsed_ms"), "wall_ms": record.get("wall_ms"),
        "status": record.get("status", "unknown"), "error": record.get("error"),
        "audit": {"source_report": source, "source_report_sha256": report_sha,
                  "source_record_key": key, "manifest_sha256": manifest_sha, "record": audit_record},
    }


def load_results(input_root):
    rows, sources, issues = [], [], []
    paths = sorted(path for path in input_root.rglob("results.json")
                   if path.relative_to(input_root).parts[0] in EXPECTED)
    for path in paths:
        label = path.relative_to(input_root).as_posix()
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            records = report["records"]
            if not isinstance(records, dict):
                raise ValueError("records must be an object keyed by job hash")
            report_sha = digest(path)
            manifest_hash = report.get("manifest_sha256", "")
            sources.append({"path": label, "sha256": report_sha,
                            "manifest_sha256": json.dumps(manifest_hash, sort_keys=True) if isinstance(manifest_hash, dict) else manifest_hash,
                            "selected_jobs": report.get("selected_jobs"), "records": len(records),
                            "errors": sum(r.get("status") != "ok" for r in records.values())})
            rows.extend(compact_record(record, key, label, report_sha,
                                       record.get("manifest_sha256", manifest_hash))
                        for key, record in records.items())
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issues.append(f"{label}: {type(exc).__name__}: {exc}")
    present = {path.relative_to(input_root).parts[0] for path in paths}
    missing = [name for name in EXPECTED if name not in present]
    return rows, sources, missing, issues


def aggregate(rows):
    groups = defaultdict(lambda: {"queries": 0, "truth": 0, "predicted": 0, "matched": 0,
                                  "missed": 0, "extra": 0, "exact": 0, "count_equal": 0, "elapsed_ms": 0.0})
    for row in rows:
        if row["status"] != "ok":
            continue
        key = (row["phase"], row["run"], row["strategy"], row["model"], row["threshold"], row["variant"],
               row["split"], row["category"])
        group = groups[key]
        group["queries"] += 1
        for field in ("truth", "predicted", "matched", "missed", "extra"):
            group[field] += row[field]
        group["exact"] += bool(row["exact"])
        group["count_equal"] += bool(row["count_equal"])
        group["elapsed_ms"] += row["elapsed_ms"] or 0
    result = []
    for key, values in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        result.append(dict(zip(("phase", "run", "strategy", "model", "threshold", "variant", "split", "category"), key))
                      | values | {"mean_ms": round(values["elapsed_ms"] / values["queries"], 1)})
    return result


def local_image(input_root, container_path):
    if not isinstance(container_path, str) or not container_path.startswith("/experiment/"):
        return None
    relative = Path(container_path.removeprefix("/experiment/"))
    if relative.is_absolute() or relative.drive or ".." in relative.parts:
        return None
    path = input_root / relative
    return path if path.is_file() else None


def thumbnail(path, missed=()):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    scale = min(1.0, 580 / max(image.width, image.height))
    if missed:
        draw = ImageDraw.Draw(image)
        radius = max(9, round(max(image.size) / 125))
        for x, y in missed:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                         outline="#ff334b", width=max(3, radius // 3))
    if scale < 1:
        image.thumbnail((580, 580), Image.Resampling.LANCZOS)
    stream = BytesIO()
    image.save(stream, format="JPEG", quality=73, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def selected_cases(rows, limit=10):
    """Choose high-error examples, with category and split coverage where available."""
    core_runs = {"baseline-tune", "mixed-tune", "heldout-core", "heldout-mixed"}
    candidates = [row for row in rows if row["status"] == "ok" and row["run"] in core_runs
                  and row["variant"] == "default" and row["threshold"] in (0.01, 0.5)]
    buckets = defaultdict(list)
    for row in candidates:
        buckets[(row["split"], row["category"])].append(row)
    chosen, seen = [], set()
    for bucket in sorted(buckets):
        choice = max(buckets[bucket], key=lambda row: (row["missed"], row["extra"], row["target_id"]))
        chosen.append(choice)
        seen.add((choice["target_sha256"], choice["category"], choice["split"]))
    for row in sorted(candidates, key=lambda row: (row["missed"], row["extra"]), reverse=True):
        if len(chosen) >= limit:
            break
        query = (row["target_sha256"], row["category"], row["split"])
        if query not in seen:
            chosen.append(row)
            seen.add(query)
    return chosen[:limit]


def case_images(rows, input_root):
    cards = []
    for row in selected_cases(rows):
        record = row["audit"]["record"]
        original = local_image(input_root, (record.get("target") or {}).get("path"))
        annotated = local_image(input_root, (record.get("prediction") or {}).get("annotated_path"))
        if original is None or annotated is None or digest(original) != row["target_sha256"]:
            continue
        points = record.get("points") or []
        missed = [points[i] for i in (record.get("metrics") or {}).get("missed_point_indices", [])]
        cards.append({"id": row["id"], "run": row["run"], "model": row["model"],
                      "target": row["target_id"], "category": row["category"], "split": row["split"],
                      "threshold": row["threshold"], "missed": row["missed"], "extra": row["extra"],
                      "original": thumbnail(original), "annotated": thumbnail(annotated, missed)})
    return cards


def spotlight(rows):
    core = {"baseline-tune", "mixed-tune", "heldout-core", "heldout-mixed"}
    baseline = [r for r in rows if r["status"] == "ok" and r["run"] == "baseline-tune"
                and r["model"] == "countgd-plusplus" and r["threshold"] == 0.5
                and r["variant"] == "default"]
    other = [r for r in rows if r["status"] == "ok" and r["run"] in core - {"baseline-tune"}
             and r["model"] == "countgd-plusplus" and r["threshold"] == 0.5
             and r["variant"] == "default"]
    return {"baseline": {"queries": len(baseline), "truth": sum(r["truth"] for r in baseline),
                          "missed": sum(r["missed"] for r in baseline),
                          "extra": sum(r["extra"] for r in baseline)} if baseline else None,
            "other": {"queries": len(other), "with_misses": sum(r["missed"] > 0 for r in other),
                      "worst": {k: max(other, key=lambda r: r["missed"])[k]
                                for k in ("target_id", "category", "split", "missed", "extra")}} if other else None}


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hardware counting reliability</title>
<style>
:root{font-family:Inter,Segoe UI,system-ui,sans-serif;color:#eaf2f5;background:#0c171f}
*{box-sizing:border-box}body{margin:0}header{padding:42px max(24px,5vw) 32px;background:linear-gradient(120deg,#173748,#10222c 65%,#17312d)}
h1{font-size:clamp(31px,4vw,54px);line-height:1.07;letter-spacing:-.04em;margin:8px 0 13px}h2{font-size:22px;margin:0 0 15px}
p{line-height:1.55}.kicker{color:#71dbbd;font-weight:700;font-size:12px;letter-spacing:.17em;text-transform:uppercase}
.intro{max-width:950px;color:#bbd0d7;margin:0}.wrap{padding:27px max(24px,5vw) 80px;max-width:1700px;margin:auto}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:0 0 26px}.card,.panel,.case{background:#14242d;border:1px solid #2b424d;border-radius:14px}
.card{padding:18px}.card strong{font-size:29px;display:block;letter-spacing:-.04em}.card span{font-size:12px;color:#a9c3cb}
.panel{padding:23px;margin:19px 0}.note{color:#b8cdd4}.warn{border-left:3px solid #f5ad66;padding:12px 16px;background:#33271f;color:#f6d7b7;border-radius:0 8px 8px 0}
.filters{display:flex;flex-wrap:wrap;gap:9px;margin-bottom:15px}select,input{background:#0d1c24;color:#eef6f7;border:1px solid #42606d;border-radius:7px;padding:9px 11px;min-width:145px}
.scroller{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th{text-align:left;color:#9fc2cb;font-weight:600;white-space:nowrap;background:#1b303a;position:sticky;top:0}th,td{padding:10px;border-bottom:1px solid #2a3f48}td{white-space:nowrap}tr:hover td{background:#1c343e}
.miss{color:#ff9ba3;font-weight:700}.good{color:#83e5bf}.muted{color:#a8bdc4}.chip{display:inline-block;background:#25414b;border-radius:6px;padding:3px 6px}
button{background:#244b55;color:#defaf0;border:1px solid #3f7280;border-radius:7px;padding:6px 9px;cursor:pointer}button:hover{background:#316171}
.cases{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}.case{padding:14px}.case h3{font-size:15px;margin:0 0 4px}.case .sub{font-size:12px;color:#b3cad0;margin:0 0 9px}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:6px}.pair img{width:100%;height:190px;object-fit:contain;background:#091218;border-radius:5px}.pair span{font-size:11px;color:#a9c2ca}
code{color:#a8ecdd}small{color:#a8bdc4}details{margin:10px 0}summary{cursor:pointer}ul{line-height:1.7}
</style></head><body>
<header><div class="kicker">Frozen-source experiment audit</div><h1>Hardware counting reliability</h1>
<p class="intro">Screws, nuts, bolts and washers across separate and mixed scenes. Every miss and extra below comes from one-to-one matching of annotated object centers to predicted boxes. Count equality alone does not establish an exact image.</p></header>
<div class="wrap"><div id="cards" class="cards"></div>
<section class="panel"><h2>Scope and interpretation</h2><div id="scope"></div></section>
<section class="panel"><h2>Comparison matrix</h2><p class="note">Each row is one run, model, threshold, reference variant, split and category. Filters update the table without combining incompatible inputs.</p>
<div class="filters" id="filters"></div><div class="scroller"><table><thead><tr><th>Phase</th><th>Run</th><th>Strategy</th><th>Model</th><th>Threshold</th><th>Reference</th><th>Split</th><th>Category</th><th>Queries</th><th>Missed</th><th>Extra</th><th>Exact</th><th>Count equal</th><th>Mean ms</th></tr></thead><tbody id="matrix"></tbody></table></div></section>
<section class="panel"><h2>Image audit</h2><p class="note">Red rings on the saved model overlay mark missed ground-truth centers for this report only. Model boxes and scores are unchanged. Original photographs are shown alongside the overlay.</p><div id="cases" class="cases"></div></section>
<section class="panel"><h2>All query records</h2><p class="note">Each download contains its saved detection boxes, ground-truth centers, matching indices, timing, hashes and settings. Errors appear here too.</p>
<div class="filters" id="recordFilters"></div><div class="scroller"><table><thead><tr><th>Phase</th><th>Run</th><th>Target</th><th>Category</th><th>Split</th><th>Model</th><th>Threshold</th><th>Reference</th><th>Result</th><th>Missed</th><th>Extra</th><th>Count equal</th><th>Audit JSON</th></tr></thead><tbody id="records"></tbody></table></div></section>
<section class="panel"><h2>Inputs and provenance</h2><div id="sources"></div></section>
</div><script type="application/json" id="payload">__PAYLOAD__</script><script>
const d=JSON.parse(document.getElementById('payload').textContent), esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=x=>x==null?'—':Number(x).toLocaleString(), percent=(a,b)=>b?`${Math.round(100*a/b)}%`:'—';
document.getElementById('cards').innerHTML=[['Study photos',d.design.all.photos],['Category queries',d.design.all.queries],['Heldout photos / queries',`${d.design.heldout.photos} / ${d.design.heldout.queries}`],['Saved query records',d.rows.length],['Saved result files',d.sources.length]].map(([l,v])=>`<div class="card"><strong>${esc(v)}</strong><span>${esc(l)}</span></div>`).join('');
const missing=d.missing.length?`<p class="warn">Pending inputs: ${esc(d.missing.join(', '))}. Tables reflect only saved reports discovered at generation time.</p>`:'';
const incomplete=d.sources.filter(s=>s.selected_jobs!=null&&s.records<s.selected_jobs).map(s=>`${s.path} ${s.records}/${s.selected_jobs}`);
const partial=incomplete.length?`<p class="warn">Reports still filling: ${esc(incomplete.join(', '))}. Metrics below use completed saved records only.</p>`:'';
const issues=d.issues.length?`<p class="warn">Unreadable or unsupported results: ${esc(d.issues.join(' · '))}</p>`:'';
const b=d.spotlight.baseline,o=d.spotlight.other;
const baseline=b?`<p>Frozen CountGD++ 0.50 baseline on ${fmt(b.queries)} tune queries: ${fmt(b.missed)} missed centers and ${fmt(b.extra)} extra boxes among ${fmt(b.truth)} annotated objects. Misses and extras are measured separately from the net predicted count.</p>`:'';
const other=o?`<p>CountGD++ 0.50 also missed objects in ${fmt(o.with_misses)} of ${fmt(o.queries)} saved mixed/heldout core queries; the largest miss count is ${fmt(o.worst.missed)} on ${esc(o.worst.target_id)} (${esc(o.worst.category)}, ${esc(o.worst.split)}), with ${fmt(o.worst.extra)} extra boxes.</p>`:'';
document.getElementById('scope').innerHTML=`${missing}${partial}${issues}<p>Designed sample: ${fmt(d.design.tune.photos)} tune photos / ${fmt(d.design.tune.queries)} category queries, and ${fmt(d.design.heldout.photos)} heldout photos / ${fmt(d.design.heldout.queries)} category queries. Repeated models and thresholds on the same photo are not independent samples.</p>${baseline}${other}<p>Center containment with maximum one-to-one matching measures object-count evidence, not full segmentation or box extent. ScrewCount provides centers for all objects but no complete target boxes; its target IoU cannot be scored. Mixed-hardware source boxes may miss tiny or occluded parts, and related video/source photos are not independent hardware lots. Fully hidden objects cannot be verified from pixels.</p><p>The SAM3 threshold sweep uses tune queries only. Server JPEG normalization is a separate heldout stress check of the frozen configurations, not a new model ranking or threshold selection; it uses re-encoded target and reference bytes with unchanged ground-truth coordinates. It does not reproduce the browser reference-canvas JPEG quality 0.94 step. Compare normalized results to raw results only with the input-byte change made explicit.</p><p>The reference-background experiment includes nuts and screws only: audited masks failed for bolt and washer references. Its gray and white variants are FastSAM-guided RGB context composites; transparent-alpha inference was not tested. Nut holes retain original wood, so these are limited context comparisons rather than clean transparent cutouts. Background variants, new references, model strategies, and different target JPEG bytes must be compared within identical target hashes. Category labels come from source datasets and may be ambiguous or incomplete. Zero-error counting is not validated; real SKUs need an audited, representative dataset and likely domain-specific training and review.</p>`;
function selector(host,rows,fields,onchange){let html='';for(const [key,label] of fields){const vals=[...new Set(rows.map(x=>String(x[key]??'')))].sort();html+=`<select data-key="${key}" aria-label="${label}"><option value="">All ${label}</option>${vals.map(v=>`<option value="${esc(v)}">${esc(v)}</option>`).join('')}</select>`}host.innerHTML=html;host.querySelectorAll('select').forEach(x=>x.addEventListener('change',onchange));}
function filtered(host,rows){const choices=[...host.querySelectorAll('select')].filter(x=>x.value).map(x=>[x.dataset.key,x.value]);return rows.filter(r=>choices.every(([k,v])=>String(r[k]??'')===v));}
const groupFilters=document.getElementById('filters');selector(groupFilters,d.groups,[['phase','phases'],['run','runs'],['model','models'],['strategy','strategies'],['category','categories'],['split','splits'],['variant','references']],drawMatrix);
function drawMatrix(){document.getElementById('matrix').innerHTML=filtered(groupFilters,d.groups).map(r=>`<tr><td>${esc(r.phase)}</td><td>${esc(r.run)}</td><td>${esc(r.strategy)}</td><td>${esc(r.model)}</td><td>${fmt(r.threshold)}</td><td>${esc(r.variant)}</td><td>${esc(r.split)}</td><td>${esc(r.category)}</td><td>${fmt(r.queries)}</td><td class="miss">${fmt(r.missed)}</td><td>${fmt(r.extra)}</td><td>${fmt(r.exact)}/${fmt(r.queries)}</td><td>${fmt(r.count_equal)}/${fmt(r.queries)}</td><td>${fmt(r.mean_ms)}</td></tr>`).join('')||'<tr><td colspan="14">No matching rows</td></tr>';}
drawMatrix();
document.getElementById('cases').innerHTML=d.cases.map(c=>`<article class="case"><h3>${esc(c.target)} · ${esc(c.category)}</h3><p class="sub">${esc(c.split)} · ${esc(c.model)} @ ${fmt(c.threshold)} · ${fmt(c.missed)} missed / ${fmt(c.extra)} extra</p><div class="pair"><div><img src="${c.original}" alt="Original ${esc(c.target)}"><span>Original</span></div><div><img src="${c.annotated}" alt="Model overlay with missed centers"><span>Prediction + red missed centers</span></div></div></article>`).join('')||'<p>No verified image pairs were available.</p>';
const recordFilters=document.getElementById('recordFilters');selector(recordFilters,d.rows,[['phase','phases'],['run','runs'],['model','models'],['category','categories'],['split','splits'],['status','statuses']],drawRecords);
function drawRecords(){document.getElementById('records').innerHTML=filtered(recordFilters,d.rows).map(r=>`<tr><td>${esc(r.phase)}</td><td>${esc(r.run)}</td><td>${esc(r.target_id)}</td><td>${esc(r.category)}</td><td>${esc(r.split)}</td><td>${esc(r.model)}</td><td>${fmt(r.threshold)}</td><td>${esc(r.variant)}</td><td>${r.status==='ok'?'<span class="good">ok</span>':`<span class="miss" title="${esc(r.error?.message||'')}">${esc(r.status)}</span>`}</td><td class="miss">${fmt(r.missed)}</td><td>${fmt(r.extra)}</td><td>${r.count_equal==null?'—':r.count_equal?'yes':'no'}</td><td><button data-id="${r.id}">Download</button></td></tr>`).join('')||'<tr><td colspan="13">No matching rows</td></tr>';}
drawRecords();document.getElementById('records').addEventListener('click',e=>{const id=e.target.dataset.id;if(!id)return;const row=d.rows.find(r=>r.id===id);if(!row)return;const a=document.createElement('a'),url=URL.createObjectURL(new Blob([JSON.stringify(row.audit,null,2)],{type:'application/json'}));a.href=url;a.download=`${row.target_id}-${row.model}-${row.id}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)});
document.getElementById('sources').innerHTML=`<p class="note">Generated ${esc(d.generated)} UTC. Source paths below are labels relative to the experiment directory; SHA-256 values bind these exact saved reports. Selected jobs counts describe the latest invocation, while records can accumulate across resumed invocations. The frozen selection files are named only if present and are not recomputed here.</p><div class="scroller"><table><thead><tr><th>Saved report</th><th>Records</th><th>Latest selected jobs</th><th>Errors</th><th>Report SHA-256</th><th>Manifest SHA-256</th></tr></thead><tbody>${d.sources.map(s=>`<tr><td>${esc(s.path)}</td><td>${fmt(s.records)}</td><td>${fmt(s.selected_jobs)}</td><td>${fmt(s.errors)}</td><td><code>${esc(s.sha256)}</code></td><td><code>${esc(s.manifest_sha256)}</code></td></tr>`).join('')}</tbody></table></div><p class="note">Frozen selections: ${esc(d.selections.join(', ')||'none found')}.</p><p class="note">Canonical analysis scripts:</p><div class="scroller"><table><thead><tr><th>Repository path</th><th>SHA-256</th></tr></thead><tbody>${d.scripts.map(s=>`<tr><td>${esc(s.path)}</td><td><code>${esc(s.sha256)}</code></td></tr>`).join('')}</tbody></table></div>`;
</script></body></html>'''


def render(input_root, output):
    rows, sources, missing, issues = load_results(input_root)
    selection_dir = ROOT / "test-results/reliability-benchmark/analysis"
    selection_paths = [selection_dir / "selection.json", selection_dir / "strategy-selection.json",
                       input_root / "background/selection-audited.json"]
    selections = [f"{path.name} (SHA-256 {digest(path)})" for path in selection_paths if path.is_file()]
    script_paths = sorted((ROOT / "scripts/reliability").rglob("*.py"))
    script_paths += [ROOT / "scripts/benchmark_reliability.py", Path(__file__)]
    scripts = [{"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)}
               for path in script_paths if path.is_file()]
    payload = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "design": study_design(), "rows": rows, "groups": aggregate(rows),
               "spotlight": spotlight(rows),
               "cases": case_images(rows, input_root), "sources": sources,
               "missing": missing, "issues": issues, "selections": selections,
               "scripts": scripts}
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(HTML.replace("__PAYLOAD__", serialized), encoding="utf-8")
    temporary.replace(output)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if not args.input_root.is_dir():
        parser.error(f"Input directory does not exist: {args.input_root}")
    payload = render(args.input_root, args.output)
    print(f"Rendered {len(payload['rows'])} records from {len(payload['sources'])} result files: {args.output}")
    if payload["issues"]:
        print("Unreadable inputs: " + "; ".join(payload["issues"]))
    return int(bool(payload["issues"]))


if __name__ == "__main__":
    raise SystemExit(main())
