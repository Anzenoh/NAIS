#!/usr/bin/env python3
"""Build an offline, method-blinded Validation caption audit from verified T004K."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scripts import build_low_rank_generation_viewer as prior_viewer
from vision_encoder_alignment.vlm_compat import query_adapter_low_rank_validation_generation as runtime
from vision_encoder_alignment.vlm_compat import query_adapter_low_rank_train_fit_runtime as fit
from vision_encoder_alignment.vlm_compat import resumable_anchor_transport as transport

PROTOCOL_PATH = Path("configs/vlm_compat/p2_low_rank_blinded_human_audit_v1.json")
PROTOCOL_SCHEMA = "vea.p2.low_rank_blinded_human_audit.v1"
ASSIGNMENT_SCHEMA = "vea.p2.low_rank_blinded_human_assignment.v1"
SCORES_SCHEMA = "vea.p2.low_rank_blinded_human_scores.v1"
T004K_SHA = "6d526c7382811cb79a78c58cd2bc6802ee793117d21ea669ff03cbcb94cddff0"
LABELS = ("A", "B", "C", "D", "E")
RUBRIC = {
    "5": "virtually all essential objects/scene/actions correct; no meaningful hallucination.",
    "4": "largely correct with only minor omissions or errors.",
    "3": "main scene broadly correct but notable mistake or omission.",
    "2": "partial match but major misunderstanding or hallucination.",
    "1": "very limited overlap; most essential content wrong.",
    "0": "main objects/scene wrong or description nearly unrelated.",
}


def load_protocol(repository: Path, path: Path | None = None) -> dict[str, Any]:
    expected_path = (repository / PROTOCOL_PATH).resolve()
    runtime.require(path is None or (path.is_absolute() and path.resolve() == expected_path),
                    "protocol path must be the reviewed repository config")
    value = runtime.read_json(expected_path)
    expected = {
        "schema_id": PROTOCOL_SCHEMA, "task_id": "VEA-P2-T004M",
        "purpose": "post_hoc_exploratory_absolute_caption_quality_not_selection",
        "partition": "validation", "image_count": 100, "condition_count": 5,
        "blinding_seed": 43, "source_t004k_result_sha256": T004K_SHA,
        "score_type": "integer_0_to_5_or_null",
        "score_reference": "actual_displayed_image_not_native_text",
        "rubric": RUBRIC, "selection_status": "NO_RANK_ACCEPTED",
    }
    runtime.require(value == expected, "frozen human-audit protocol or rubric differs")
    return value


def mapping_payload(ids: Sequence[str], protocol: Mapping[str, Any]) -> dict[str, Any]:
    runtime.require(ids and len(set(ids)) == len(ids) and
                    all(isinstance(image_id, str) and image_id for image_id in ids) and
                    protocol["blinding_seed"] == 43, "blinding image IDs or seed differ")
    entries = []
    for image_id in ids:
        seed = int.from_bytes(hashlib.sha256(
            f"43\x00{image_id}".encode("utf-8")).digest()[:16], "big")
        shuffled = list(runtime.LANES)
        random.Random(seed).shuffle(shuffled)
        entries.append({"image_id": image_id,
                        "display_to_lane": dict(zip(LABELS, shuffled, strict=True))})
    return {"ordered_image_ids": list(ids), "assignments": entries,
            "source_t004k_result_sha256": protocol["source_t004k_result_sha256"]}


def make_assignment(ids: Sequence[str], protocol: Mapping[str, Any]) -> dict[str, Any]:
    payload = mapping_payload(ids, protocol)
    return {"schema_id": ASSIGNMENT_SCHEMA, **payload,
            "protocol_sha256": runtime.digest(protocol),
            "rubric_sha256": runtime.digest(protocol["rubric"]),
            "mapping_sha256": runtime.digest(payload)}


def validate_assignment(value: Mapping[str, Any], ids: Sequence[str],
                        protocol: Mapping[str, Any]) -> None:
    expected = make_assignment(ids, protocol)
    runtime.require(isinstance(value, dict) and value == expected and
                    len(value["assignments"]) == len(ids) and
                    all(tuple(entry["display_to_lane"]) == LABELS and
                        set(entry["display_to_lane"].values()) == set(runtime.LANES)
                        for entry in value["assignments"]),
                    "assignment identity, hash, order, or bijection differs")


def empty_scores(assignment: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_id": SCORES_SCHEMA, "split": "validation",
        "source_t004k_result_sha256": assignment["source_t004k_result_sha256"],
        "rubric_sha256": assignment["rubric_sha256"],
        "mapping_sha256": assignment["mapping_sha256"],
        "ordered_image_ids": list(assignment["ordered_image_ids"]),
        "entries": [{"image_id": image_id, "ratings": {label: None for label in LABELS},
                     "notes": "", "bookmark": False}
                    for image_id in assignment["ordered_image_ids"]],
    }


def validate_scores(value: Mapping[str, Any], assignment: Mapping[str, Any],
                    protocol: Mapping[str, Any]) -> dict[str, Any]:
    ids = tuple(assignment["ordered_image_ids"])
    validate_assignment(assignment, ids, protocol)
    template = empty_scores(assignment)
    runtime.require(isinstance(value, dict) and set(value) == set(template) and
                    all(value.get(key) == template[key] for key in template if key != "entries"),
                    "score export identity, schema, split, or keys differ")
    entries = value.get("entries")
    runtime.require(isinstance(entries, list) and len(entries) == len(ids),
                    "score export image count differs")
    coverage = {label: 0 for label in LABELS}
    rated_images = 0
    for index, (entry, image_id) in enumerate(zip(entries, ids, strict=True)):
        runtime.require(isinstance(entry, dict) and
                        set(entry) == {"image_id", "ratings", "notes", "bookmark"} and
                        entry["image_id"] == image_id and
                        type(entry["notes"]) is str and type(entry["bookmark"]) is bool,
                        f"score export image order, ID, or fields differ at {index}")
        ratings = entry["ratings"]
        runtime.require(isinstance(ratings, dict) and set(ratings) == set(LABELS),
                        "score export anonymous rating keys differ")
        scored = 0
        for label in LABELS:
            score = ratings[label]
            runtime.require(score is None or (type(score) is int and 0 <= score <= 5),
                            "score must be integer 0..5 or null")
            if score is not None:
                coverage[label] += 1
                scored += 1
        rated_images += scored == len(LABELS)
    total = sum(coverage.values())
    return {"rated_slots": total, "total_slots": len(ids) * len(LABELS),
            "complete_images": rated_images, "image_count": len(ids),
            "by_anonymous_id": coverage, "complete": total == len(ids) * len(LABELS)}


def render_blind_html(rows: Sequence[Mapping[str, Any]], images: Sequence[runtime.ValidationImage],
                      assignment: Mapping[str, Any], protocol: Mapping[str, Any]) -> str:
    ids = tuple(image.image_id for image in images)
    validate_assignment(assignment, ids, protocol)
    runtime.require(len(rows) == len(ids) * len(LABELS) and
                    tuple((row["image_id"], row["lane_key"]) for row in rows) ==
                    tuple((image_id, lane) for image_id in ids for lane in runtime.LANES),
                    "verified row order differs")
    by_key = {(row["image_id"], row["lane_key"]): row for row in rows}
    payload = []
    for image, entry in zip(images, assignment["assignments"], strict=True):
        runtime.require(image.path.is_file() and runtime.sha_file(image.path) == image.sha256,
                        "Validation JPEG differs before blind embed")
        contents = image.path.read_bytes()
        runtime.require(runtime.sha_file(image.path) == image.sha256,
                        "Validation JPEG differs during blind embed")
        captions = [{"id": label, "text": by_key[(image.image_id, entry["display_to_lane"][label])]["decoded_text"]}
                    for label in LABELS]
        payload.append({"image_id": image.image_id,
                        "image": "data:image/jpeg;base64," + base64.b64encode(contents).decode("ascii"),
                        "captions": captions})
    data = {
        "schema_id": SCORES_SCHEMA, "split": "validation",
        "source_t004k_result_sha256": assignment["source_t004k_result_sha256"],
        "rubric_sha256": assignment["rubric_sha256"],
        "mapping_sha256": assignment["mapping_sha256"],
        "ordered_image_ids": list(ids), "rubric": protocol["rubric"], "images": payload,
    }
    serialized = json.dumps(data, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    serialized = serialized.replace("<", "\\u003c").replace(">", "\\u003e")
    return """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Blinded Validation image-description audit</title>
<style>
body{font:16px system-ui,sans-serif;max-width:1150px;margin:1rem auto;padding:0 1rem;color:#182431;background:#f7f9fb}
.bar{display:flex;gap:.6rem;align-items:center;flex-wrap:wrap;margin:1rem 0}
button,input,select,textarea{font:inherit;padding:.35rem}
img{max-width:min(100%,500px);max-height:420px;object-fit:contain;background:white}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:.7rem}
.card{border:1px solid #bec9d5;border-radius:.5rem;background:white;padding:.7rem}
.card p{white-space:pre-wrap;overflow-wrap:anywhere}
.rubric{padding:.6rem 1rem;background:#e9f1f8}
textarea{width:95%}
</style>
<h1>Blinded Validation image-description audit</h1>
<p>Rate each description against the displayed image. The score measures image-grounded accuracy.
An empty description receives the same rubric. Unrated is distinct from 0. This is an exploratory audit.</p>
<section class="rubric"><h2>Fixed 0–5 accuracy rubric</h2><ol reversed start="5" id="rubric"></ol></section>
<div class="bar"><button id="prev">Previous</button><button id="next">Next</button>
<input id="filter" type="search" aria-label="Filter image ID" placeholder="Filter image ID">
<label><input id="onlyUnrated" type="checkbox">Incomplete images only</label>
<button id="bookmark">Bookmark</button><span id="position"></span><span id="progress"></span></div>
<h2 id="imageId"></h2><img id="photo" alt="Validation image">
<div id="captions" class="grid"></div>
<label>Optional notes for this image<textarea id="notes" rows="3"></textarea></label>
<div class="bar"><button id="export">Export score backup JSON</button>
<label>Import score backup JSON <input id="import" type="file" accept="application/json,.json"></label></div>
<p id="message" role="status"></p>
<script>
"use strict";
const data=__DATA__;
const labels=["A","B","C","D","E"];
const storageKey="vea-t004m-blind-validation-v1-"+data.source_t004k_result_sha256+"-"+data.mapping_sha256;
const $=id=>document.getElementById(id);
const exactKeys=(value,keys)=>value&&typeof value==="object"&&!Array.isArray(value)&&
  Object.keys(value).sort().join("|")===keys.slice().sort().join("|");
const empty=()=>({schema_id:data.schema_id,split:data.split,
  source_t004k_result_sha256:data.source_t004k_result_sha256,
  rubric_sha256:data.rubric_sha256,mapping_sha256:data.mapping_sha256,
  ordered_image_ids:data.ordered_image_ids.slice(),
  entries:data.images.map(item=>({image_id:item.image_id,
    ratings:Object.fromEntries(labels.map(label=>[label,null])),notes:"",bookmark:false}))});
function validate(value){
  const reference=empty();
  if(!exactKeys(value,Object.keys(reference))||value.schema_id!==reference.schema_id||
     value.split!=="validation"||
     value.source_t004k_result_sha256!==reference.source_t004k_result_sha256||
     value.rubric_sha256!==reference.rubric_sha256||
     value.mapping_sha256!==reference.mapping_sha256||
     !Array.isArray(value.ordered_image_ids)||
     JSON.stringify(value.ordered_image_ids)!==JSON.stringify(reference.ordered_image_ids)||
     !Array.isArray(value.entries)||value.entries.length!==reference.entries.length)
    throw Error("Score backup identity or image count differs");
  value.entries.forEach((entry,index)=>{
    if(!exactKeys(entry,["image_id","ratings","notes","bookmark"])||
       entry.image_id!==reference.entries[index].image_id||
       typeof entry.notes!=="string"||typeof entry.bookmark!=="boolean"||
       !exactKeys(entry.ratings,labels))
      throw Error("Score backup image ID, order, or fields differ");
    labels.forEach(label=>{const score=entry.ratings[label];
      if(score!==null&&(!Number.isInteger(score)||score<0||score>5))
        throw Error("Score must be 0–5 or Unrated")})});
  return value}
let scores=empty();
try{const previous=localStorage.getItem(storageKey);
  if(previous)scores=validate(JSON.parse(previous))}catch(_error){scores=empty()}
function persist(){try{localStorage.setItem(storageKey,JSON.stringify(scores))}
  catch(_error){$("message").textContent="Browser storage unavailable; export a score backup"}}
function coverage(){let total=0,complete=0;
  for(const entry of scores.entries){let n=0;
    for(const label of labels)if(entry.ratings[label]!==null)n++;
    total+=n;if(n===5)complete++}
  return {total,complete}}
function eligible(){const term=$("filter").value.toLowerCase();
  return data.images.map((item,index)=>({item,index})).filter(({item,index})=>
    item.image_id.toLowerCase().includes(term)&&
    (!$("onlyUnrated").checked||labels.some(label=>scores.entries[index].ratings[label]===null)))}
let position=0;
for(const score of ["5","4","3","2","1","0"]){const li=document.createElement("li");
  li.value=Number(score);li.textContent=data.rubric[score];$("rubric").append(li)}
function render(){const matches=eligible();position=Math.min(position,Math.max(0,matches.length-1));
  const count=coverage();$("progress").textContent=count.total+" / "+(data.images.length*5)+
    " ratings; "+count.complete+" / "+data.images.length+" images complete"+
    (count.total===data.images.length*5?"":" — INCOMPLETE");
  $("captions").replaceChildren();
  if(!matches.length){$("imageId").textContent="No matching images";$("photo").removeAttribute("src");
    $("position").textContent="0 / 0";return}
  const {item,index}=matches[position],entry=scores.entries[index];
  $("imageId").textContent=item.image_id;$("photo").src=item.image;
  $("position").textContent=(position+1)+" / "+matches.length;
  $("bookmark").textContent=entry.bookmark?"Remove bookmark":"Bookmark";
  $("notes").value=entry.notes;
  for(const caption of item.captions){const card=document.createElement("section");card.className="card";
    const h=document.createElement("h3");h.textContent="Caption "+caption.id;card.append(h);
    const p=document.createElement("p");p.textContent=caption.text;card.append(p);
    const label=document.createElement("label");label.textContent="Image accuracy: ";
    const select=document.createElement("select");
    for(const optionValue of ["","0","1","2","3","4","5"]){const option=document.createElement("option");
      option.value=optionValue;option.textContent=optionValue===""?"Unrated":optionValue;
      select.append(option)}
    select.value=entry.ratings[caption.id]===null?"":String(entry.ratings[caption.id]);
    select.onchange=()=>{entry.ratings[caption.id]=select.value===""?null:Number(select.value);
      persist();render()};label.append(select);card.append(label);$("captions").append(card)}}
$("prev").onclick=()=>{position=Math.max(0,position-1);render()};
$("next").onclick=()=>{position=Math.min(eligible().length-1,position+1);render()};
$("filter").oninput=()=>{position=0;render()};
$("onlyUnrated").onchange=()=>{position=0;render()};
$("bookmark").onclick=()=>{const item=eligible()[position];if(!item)return;
  const entry=scores.entries[item.index];entry.bookmark=!entry.bookmark;persist();render()};
$("notes").oninput=()=>{const item=eligible()[position];if(!item)return;
  scores.entries[item.index].notes=$("notes").value;persist()};
$("export").onclick=()=>{const value=validate(scores),blob=new Blob([JSON.stringify(value,null,2)],
  {type:"application/json"}),link=document.createElement("a");link.href=URL.createObjectURL(blob);
  link.download="vea-t004m-blinded-validation-scores.json";link.click();
  setTimeout(()=>URL.revokeObjectURL(link.href),1000);
  $("message").textContent=coverage().total===data.images.length*5?"Complete backup exported":
    "Partial backup exported — INCOMPLETE_NO_FINAL_INFERENCE"};
$("import").onchange=async event=>{const file=event.target.files[0];if(!file)return;
  try{const candidate=validate(JSON.parse(await file.text()));scores=candidate;persist();
    $("message").textContent="Score backup imported";position=0;render()}
  catch(error){$("message").textContent="Import refused: "+error.message}
  event.target.value=""};
render();
</script></html>""".replace("__DATA__", serialized)


def output_directory(value: Path, roots: Sequence[Path]) -> Path:
    runtime.require(value.is_absolute() and value.name not in ("", ".", "..") and
                    value.parent.is_dir() and not value.exists() and not value.is_symlink(),
                    "blind output root must be fresh, absolute, and absent")
    output = value.parent.resolve(strict=True) / value.name
    runtime.require(all(not output.is_relative_to(root) and not root.is_relative_to(output)
                        for root in roots), "blind output root overlaps an input")
    return output


def build(*, repository: Path, result_root: Path, result_sha256: str,
          t001d_root: Path, output_root: Path) -> dict[str, Any]:
    protocol = load_protocol(repository)
    runtime.require(result_sha256 == protocol["source_t004k_result_sha256"],
                    "T004K reviewed result SHA differs")
    result, rows, images = prior_viewer.load_verified(
        repository=repository, result_root=result_root, result_sha256=result_sha256,
        t001d_root=t001d_root)
    ids = tuple(result["ordered_image_ids"])
    runtime.require(len(ids) == protocol["image_count"] == 100 and
                    len(rows) == 500 and len(images) == 100 and
                    result["selection_status"] == "NO_RANK_ACCEPTED" and
                    result["test_generation_count"] == 0 and
                    result["synthetic_test_only"] is False, "T004K production identity differs")
    assignment = make_assignment(ids, protocol)
    validate_assignment(assignment, ids, protocol)
    plan, _ = transport.load_bound_plan(t001d_root)
    dataset_root = Path(plan["roots"]["dataset_root"]).resolve()
    output = output_directory(output_root, (repository, result_root, t001d_root, dataset_root))
    html = render_blind_html(rows, images, assignment, protocol).encode("utf-8")
    mapping = runtime.canonical(assignment)
    output.mkdir()
    runtime.write_new(output / "blinded_viewer.html", html)
    runtime.write_new(output / "assignment.json", mapping)
    runtime.require(runtime.sha_file(output / "blinded_viewer.html") == hashlib.sha256(html).hexdigest()
                    and runtime.sha_file(output / "assignment.json") == hashlib.sha256(mapping).hexdigest(),
                    "published blind artifacts changed")
    return {"status": "PASS_BLINDED_VIEWER_BUILD", "image_count": 100,
            "anonymous_slots": 500, "source_t004k_result_sha256": result_sha256,
            "mapping_sha256": assignment["mapping_sha256"],
            "html_sha256": runtime.sha_file(output / "blinded_viewer.html"),
            "assignment_sha256": runtime.sha_file(output / "assignment.json")}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    for name in ("repository-root", "result-root", "t001d-run-root", "output-root"):
        value.add_argument(f"--{name}", type=Path, required=True)
    value.add_argument("--result-sha256", required=True)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        repository = fit._input_root(args.repository_root, "repository_root")
        runtime.require(repository == ROOT.resolve(), "repository root differs")
        result_root = fit._input_root(args.result_root, "result_root")
        t001d_root = fit._input_root(args.t001d_run_root, "t001d_run_root")
        result = build(repository=repository, result_root=result_root,
                       result_sha256=args.result_sha256, t001d_root=t001d_root,
                       output_root=args.output_root)
    except (runtime.DiagnosticError, OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(json.dumps({"status": "BLOCKED", "message": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
