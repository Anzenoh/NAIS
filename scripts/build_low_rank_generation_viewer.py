#!/usr/bin/env python3
"""Build a local, read-only T004J Validation comparison and optional scoring viewer."""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_encoder_alignment.vlm_compat import query_adapter_low_rank_validation_generation as runtime
from vision_encoder_alignment.vlm_compat import query_adapter_low_rank_train_fit_runtime as fit
from vision_encoder_alignment.vlm_compat import resumable_anchor_transport as transport

STORAGE_KEY = "vea-t004j-exploratory-validation-ratings-v1"


def render_html(rows: Sequence[Mapping[str, Any]], images: Sequence[runtime.ValidationImage],
                result: Mapping[str, Any]) -> str:
    """Render one self-contained HTML file; payload is JSON-escaped before entering JS."""
    by_key = {(row["image_id"], row["lane_key"]): row for row in rows}
    payload = []
    for image in images:
        require = runtime.require
        require(image.path.is_file() and runtime.sha_file(image.path) == image.sha256,
                "Validation JPEG changed before viewer embed")
        image_bytes = image.path.read_bytes()
        require(runtime.sha_file(image.path) == image.sha256, "Validation JPEG changed during viewer embed")
        lanes = []
        for key in runtime.LANES:
            row = by_key[(image.image_id, key)]
            lanes.append({"key": key, "text": row["decoded_text"], "tokens": row["token_ids"],
                          "f1": row["token_unigram_f1"], "exact": row["native_exact_match"],
                          "same_image_rank": row["same_image_rank"]})
        payload.append({"image_id": image.image_id, "relative_path": image.relative_path,
                        "image": "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("ascii"),
                        "lanes": lanes})
    data = {"study_id": runtime.SCHEMA, "role": result["role"], "split": "validation",
            "selection_status": "NO_RANK_ACCEPTED", "result_sha256": result["result_sha256"],
            "images": payload}
    serialized = json.dumps(data, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    serialized = serialized.replace("<", "\\u003c").replace(">", "\\u003e")
    return """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>T004J exploratory Validation comparison</title>
<style>
body{font:16px system-ui,sans-serif;margin:1rem auto;max-width:1350px;color:#17212b;background:#f8fafc}
header,.bar{display:flex;gap:.7rem;align-items:center;flex-wrap:wrap}
button,select,input,textarea{font:inherit;padding:.3rem}.bar{margin:1rem 0}
#photo{max-width: min(100%,480px);max-height:420px;object-fit:contain;background:white}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:.7rem}
.lane{border:1px solid #ccd5df;border-radius:.4rem;background:white;padding:.7rem;min-width:0}
.lane h3{font-size:1rem;margin:0}.lane p{white-space:pre-wrap;overflow-wrap:anywhere}
.meta{font-size:.85rem;color:#425466}.pair{border-top:2px solid #5c728b;margin-top:.8rem}
</style>
<header><h1>T004J exploratory Validation comparison</h1></header>
<p>Descriptive native-output comparison only. Ratings are optional and are not a rank selector or a measure of independent answer correctness. No historical scoring rubric is assumed.</p>
<div class="bar"><button id="prev">Previous</button><button id="next">Next</button>
<input id="filter" type="search" aria-label="Filter image ID" placeholder="Filter image ID">
<label><input id="onlyBookmarks" type="checkbox">Bookmarks only</label>
<button id="bookmark">Bookmark</button><span id="position"></span></div>
<h2 id="imageId"></h2><p id="path" class="meta"></p><img id="photo" alt="Validation image">
<div id="lanes" class="grid"></div>
<p><button id="export">Export ratings and notes JSON</button></p>
<script>
"use strict";
const dataset=__DATA__;
const storageKey="__STORAGE_KEY__";
let saved={};
try{const parsed=JSON.parse(localStorage.getItem(storageKey)||"{}");
  if(parsed && parsed.schema_id==="vea.p2.t004j.viewer_scores.v1" &&
     parsed.result_sha256===dataset.result_sha256 && typeof parsed.entries==="object")
    saved=parsed.entries;
}catch(_error){}
const $=id=>document.getElementById(id);
let position=0;
function entry(id){const raw=saved[id],value=raw&&typeof raw==="object"?raw:{};
  const ratings={};for(const lane of dataset.images[0].lanes){
    const score=value.ratings&&value.ratings[lane.key];
    ratings[lane.key]=Number.isInteger(score)&&score>=0&&score<=5?score:null}
  return {bookmark:value.bookmark===true,ratings,
    notes:typeof value.notes==="string"?value.notes:""}}
function persist(){localStorage.setItem(storageKey,JSON.stringify({
  schema_id:"vea.p2.t004j.viewer_scores.v1",result_sha256:dataset.result_sha256,
  study_id:dataset.study_id,split:"validation",entries:saved
}))}
function eligible(){const term=$("filter").value.toLowerCase();
  return dataset.images.filter(x=>x.image_id.toLowerCase().includes(term) &&
    (!$("onlyBookmarks").checked || entry(x.image_id).bookmark))}
function render(){const matches=eligible();if(position>=matches.length)position=Math.max(0,matches.length-1);
  $("lanes").replaceChildren();
  if(!matches.length){$("imageId").textContent="No images";$("path").textContent="";
    $("photo").removeAttribute("src");$("position").textContent="0 / 0";return}
  const item=matches[position],state=entry(item.image_id);
  $("imageId").textContent=item.image_id;$("path").textContent=item.relative_path;
  $("photo").src=item.image;$("position").textContent=(position+1)+" / "+matches.length;
  $("bookmark").textContent=state.bookmark?"Remove bookmark":"Bookmark";
  for(const lane of item.lanes){const card=document.createElement("section");card.className="lane";
    const title=document.createElement("h3");title.textContent=lane.key;card.append(title);
    const text=document.createElement("p");text.textContent=lane.text;card.append(text);
    const meta=document.createElement("p");meta.className="meta";
    meta.textContent=lane.key==="NATIVE"?"Native reference":
      "Token F1 "+lane.f1.toFixed(3)+" · exact "+lane.exact+
      " · same-image rank "+lane.same_image_rank;card.append(meta);
    const label=document.createElement("label");label.textContent="Optional rating (0–5): ";
    const select=document.createElement("select");for(const value of ["","0","1","2","3","4","5"]){
      const option=document.createElement("option");option.value=value;
      option.textContent=value===""?"Unrated":value;select.append(option)}
    select.value=state.ratings[lane.key]??"";
    select.onchange=()=>{const next=entry(item.image_id);
      next.ratings[lane.key]=select.value===""?null:Number(select.value);
      saved[item.image_id]=next;persist()};label.append(select);card.append(label);
    $("lanes").append(card)}
  const notes=document.createElement("label");notes.textContent="Notes for this image: ";
  const area=document.createElement("textarea");area.rows=3;area.style.width="95%";
  area.value=state.notes||"";area.oninput=()=>{const next=entry(item.image_id);
    next.notes=area.value;saved[item.image_id]=next;persist()};notes.append(area);
  $("lanes").append(notes)}
$("prev").onclick=()=>{position=Math.max(0,position-1);render()};
$("next").onclick=()=>{position=Math.min(eligible().length-1,position+1);render()};
$("filter").oninput=()=>{position=0;render()};
$("onlyBookmarks").onchange=()=>{position=0;render()};
$("bookmark").onclick=()=>{const item=eligible()[position];if(!item)return;
  const next=entry(item.image_id);next.bookmark=!next.bookmark;
  saved[item.image_id]=next;persist();render()};
$("export").onclick=()=>{const entries=dataset.images.map(item=>{
  const value=entry(item.image_id);
  const ratings=Object.fromEntries(item.lanes.map(lane=>[lane.key,value.ratings[lane.key]??null]));
  return {image_id:item.image_id,bookmark:!!value.bookmark,ratings,notes:value.notes||""}});
  const value={schema_id:"vea.p2.t004j.viewer_scores.v1",study_id:dataset.study_id,
    result_sha256:dataset.result_sha256,split:"validation",
    purpose:"post_hoc_exploratory_not_rank_selection",rating_range:[0,5],
    entries};const blob=new Blob([JSON.stringify(value,null,2)],{type:"application/json"});
  const link=document.createElement("a");link.href=URL.createObjectURL(blob);
  link.download="vea-t004j-validation-ratings.json";link.click();
  setTimeout(()=>URL.revokeObjectURL(link.href),1000)};
render();
</script></html>""".replace("__STORAGE_KEY__", STORAGE_KEY).replace("__DATA__", serialized)


def fresh_html_path(output_file: Path, roots: Sequence[Path]) -> Path:
    output = output_file.resolve()
    runtime.require(output_file.is_absolute() and not output_file.is_symlink() and
                    output.parent.is_dir() and not output.exists() and
                    all(not output.is_relative_to(root) and not root.is_relative_to(output)
                        for root in roots) and output.suffix.lower() == ".html",
                    "viewer output must be a fresh external HTML file")
    return output


def build(*, repository: Path, result_root: Path, result_sha256: str,
          t001d_root: Path, output_file: Path) -> Path:
    config = runtime.load_config(repository)
    result_path = result_root / "result.json"
    runtime.require(len(result_sha256) == 64 and
                    all(c in "0123456789abcdef" for c in result_sha256) and
                    runtime.sha_file(result_path) == result_sha256, "T004J result SHA differs")
    result = runtime.read_json(result_path)
    runtime.require(set(result) == runtime.RESULT_KEYS, "T004J result has unsupported fields")
    required = {"schema_id": runtime.RESULT_SCHEMA, "status": runtime.STATUS,
                "task_id": "VEA-P2-T004J", "study_id": runtime.SCHEMA,
                "role": config["role"], "split": "validation",
                "selection_status": "NO_RANK_ACCEPTED", "synthetic_test_only": False,
                "validation_image_count": 100, "lane_order": list(runtime.LANES),
                "generation_counts": {"native": 100, "adapted": 400, "total": 500},
                "config_sha256": runtime.digest(config),
                "t004f_result_sha256": runtime.T004F_SHA,
                "t004h_result_sha256": runtime.T004H_SHA, **runtime.ZERO}
    for key, value in required.items():
        runtime.require(result.get(key) == value and type(result.get(key)) is type(value),
                        f"T004J result {key} differs")
    ids = tuple(result.get("ordered_image_ids", ()))
    runtime.require(len(ids) == 100 and len(set(ids)) == 100 and
                    all(isinstance(item, str) and item for item in ids), "T004J IDs invalid")
    rows_path, aggregate_path = result_root / "rows.jsonl", result_root / "aggregate.json"
    runtime.require(runtime.sha_file(rows_path) == result.get("rows_sha256") and
                    runtime.sha_file(aggregate_path) == result.get("aggregate_sha256"),
                    "T004J rows or aggregate SHA differs")
    images = runtime.validation_images(t001d_root, ids, result["provenance"])
    rows = runtime.read_rows(rows_path)
    hashes = result["artifact_sha256"]
    ranks = {source: {method: {} for method in ("OLS_REFERENCE", "LOW_RANK_256")}
             for source in runtime.binding.SOURCE_ORDER}
    for row in rows:
        if row.get("lane_key") in runtime.LANES[1:]:
            ranks[row["source"]][row["method"]][row["image_id"]] = row.get("same_image_rank")
    summary = runtime.validate_rows(rows, ids, images, hashes, ranks, config)
    runtime.require(summary == runtime.read_json(aggregate_path) == result.get("aggregate"),
                    "T004J aggregate differs")
    plan, _ = transport.load_bound_plan(t001d_root)
    dataset_root = Path(plan["roots"]["dataset_root"]).resolve()
    output = fresh_html_path(output_file, (repository, result_root, t001d_root, dataset_root))
    html = render_html(rows, images, {**result, "result_sha256": result_sha256})
    runtime.write_new(output, html.encode("utf-8"))
    return output


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--repository-root", type=Path, required=True)
    value.add_argument("--result-root", type=Path, required=True)
    value.add_argument("--result-sha256", required=True)
    value.add_argument("--t001d-run-root", type=Path, required=True)
    value.add_argument("--output-file", type=Path, required=True)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        repository = fit._input_root(args.repository_root, "repository_root")
        runtime.require(repository == ROOT.resolve(), "repository root differs")
        result_root = fit._input_root(args.result_root, "result_root")
        t001d = fit._input_root(args.t001d_run_root, "t001d_run_root")
        output = build(repository=repository, result_root=result_root,
                       result_sha256=args.result_sha256, t001d_root=t001d,
                       output_file=args.output_file)
    except (runtime.DiagnosticError, OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(json.dumps({"status": "BLOCKED", "message": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps({"status": "PASS_OFFLINE_VIEWER", "output_file": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
