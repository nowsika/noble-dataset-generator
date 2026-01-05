#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_generate.py

- Reads scenario_template.json + scenarios.txt
- Injects qa.language + qa.metaphor_pack + (optional) forced metaphor_domain rotation
- Calls OpenAI Responses API to generate one JSON object per scenario
- Validates with jsonschema + taxonomies + language + cross-domain metaphor term warnings
- One optional retry on QA fail or style_warnings
- One optional JSON-repair call if model output is not valid JSON

Environment variables:
  OPENAI_MODEL               (default: gpt-5.2)
  TEMPLATE_PATH              (default: scenario_template.json)
  SCENARIOS_PATH             (default: scenarios.txt)
  OUTPUT_PATH                (default: outputs.jsonl)
  FORCE_ROTATE_DOMAINS       (default: true)
  REASONING_EFFORT           (default: low)   # low|medium|high
  RETRY_ON_STYLE_WARNINGS    (default: true)
  MAX_RETRY                  (default: 1)
  BAD_OUTPUT_DIR             (default: bad_outputs)
"""

import json
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from jsonschema import validate
from jsonschema.exceptions import ValidationError

from prompt_builder import build_instructions
from auto_inject import (
    ALLOWED_DOMAINS,
    detect_language,
    parse_scenario_prefix,
    inject_qa_defaults,
)
from cs_presets import apply_cs_presets

# -----------------------
# Config
# -----------------------
load_dotenv()
client = OpenAI()

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")

TEMPLATE_PATH = os.getenv("TEMPLATE_PATH", "scenario_template.json")
SCENARIOS_PATH = os.getenv("SCENARIOS_PATH", "scenarios.txt")
OUT_PATH = os.getenv("OUTPUT_PATH", "outputs.jsonl")

FORCE_ROTATE_DOMAINS = os.getenv("FORCE_ROTATE_DOMAINS", "true").lower() in ("1", "true", "yes")
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "low").strip().lower() or "low"

# Dataset profile
# - noble (default): counseling-flavored template/prompt set
# - cs: B2B SaaS Customer Support dataset (business tags/signals + CS prompt profile)
DATASET_PROFILE = (os.getenv("DATASET_PROFILE", "noble") or "noble").strip().lower()

RETRY_ON_STYLE_WARNINGS = os.getenv("RETRY_ON_STYLE_WARNINGS", "true").lower() in ("1", "true", "yes")
MAX_RETRY = int(os.getenv("MAX_RETRY", "1"))
BAD_OUTPUT_DIR = os.getenv("BAD_OUTPUT_DIR", "bad_outputs")

# -----------------------
# Utils
# -----------------------
from copy import deepcopy

def merge_additions_preserve_original(original: dict, generated: dict) -> tuple[dict, list[str]]:
    """
    - original의 기존 키/값은 절대 덮어쓰지 않는다(=preserve).
    - generated가 새로 추가한 키만 original에 병합한다.
    - dict 안쪽도 재귀적으로 "새 키만" 병합한다. (qa.style_warnings 같은 케이스)
    - generated가 original의 기존 키를 바꾸려 하면 override 경고를 남긴다.
    """
    overrides: list[str] = []
    out = deepcopy(original)

    def _merge(o, g, path=""):
        nonlocal overrides
        if not isinstance(o, dict) or not isinstance(g, dict):
            return

        for k, gv in g.items():
            p = f"{path}.{k}" if path else k

            if k not in o:
                o[k] = gv
                continue

            ov = o[k]
            # 둘 다 dict면 안쪽으로 들어가서 "새 키만" 추가
            if isinstance(ov, dict) and isinstance(gv, dict):
                _merge(ov, gv, p)
            else:
                # 원본 키를 바꾸려는 시도 = 무시 + 경고
                if gv != ov:
                    overrides.append(f"OVERRIDE_IGNORED[{p}]")
        return

    _merge(out, generated)
    return out, overrides

def read_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def read_scenarios_txt(path: str) -> List[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    out: List[str] = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out

def append_jsonl(path: str, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def save_bad_output(idx: int, text: str, suffix: str) -> None:
    ensure_dir(BAD_OUTPUT_DIR)
    p = Path(BAD_OUTPUT_DIR) / f"bad_output_{idx:05d}{suffix}.txt"
    p.write_text(text, encoding="utf-8")

def try_parse_json(text: str) -> Optional[dict]:
    """
    Best-effort parse:
      1) json.loads(text)
      2) try parse substring from first '{' to last '}'
    """
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    chunk = text[start : end + 1]
    try:
        obj = json.loads(chunk)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

def coerce_house_style_strings(out_obj: dict) -> None:
    """
    Fix common schema violations where model outputs dicts instead of strings.
    Mutates out_obj in-place.
    Applies to model_response_B and model_response_C.
    """
    for key in ("model_response_B", "model_response_C"):
        b = out_obj.get(key)
        if not isinstance(b, dict):
            continue

        # anchor dict -> string
        anchor = b.get("anchor")
        if isinstance(anchor, dict):
            title = anchor.get("title") or anchor.get("step")
            detail = anchor.get("detail")
            if title and detail:
                b["anchor"] = f"{title}: {detail}"
            elif detail:
                b["anchor"] = str(detail)
            elif title:
                b["anchor"] = str(title)
            else:
                b["anchor"] = ""

        # route_steps: enforce list[str]
        rs = b.get("route_steps")
        if isinstance(rs, list):
            fixed: List[str] = []
            for step in rs:
                if isinstance(step, dict):
                    title = step.get("step") or step.get("title")
                    detail = step.get("detail")
                    if title and detail:
                        fixed.append(f"{title}: {detail}")
                    elif detail:
                        fixed.append(str(detail))
                    elif title:
                        fixed.append(str(title))
                    else:
                        fixed.append("")
                else:
                    fixed.append(str(step))
            b["route_steps"] = fixed
def collect_text(out_obj: dict) -> str:
    chunks: List[str] = []

    a = out_obj.get("model_response_A", {})
    if isinstance(a, dict):
        chunks.append(str(a.get("raw_essay", "")))
        chunks.append(str(a.get("micro_line", "")))

    for key in ("model_response_B", "model_response_C"):
        b = out_obj.get(key, {})
        if isinstance(b, dict):
            for k in ("reflection", "anchor", "micro_script", "closing", "one_question"):
                chunks.append(str(b.get(k, "")))
            rs = b.get("route_steps", [])
            if isinstance(rs, list):
                chunks.extend([str(x) for x in rs])

    return "\n".join(chunks)

# -----------------------
# Language validation (heuristic)
# -----------------------
_CJK_RE = re.compile(r"[가-힣ㄱ-ㅎㅏ-ㅣぁ-んァ-ン一-龯]")
_JA_RE = re.compile(r"[ぁ-んァ-ン]")
_ZH_RE = re.compile(r"[\u4e00-\u9fff]")
_AR_RE = re.compile(r"[\u0600-\u06FF]")

def validate_language(text: str, expected_lang: str) -> bool:
    """
    Cheap, explainable checks (no ML).
    - en/es: reject CJK or Arabic
    - ja: must contain Japanese kana or common CJK
    - zh-*: must contain Han characters
    - ko: must contain Hangul
    - ar: must contain Arabic script
    """
    lang = (expected_lang or "").lower()

    if lang in ("en", "es"):
        return _CJK_RE.search(text) is None and _AR_RE.search(text) is None
    if lang == "ko":
        return re.search(r"[가-힣]", text) is not None
    if lang == "ja":
        return _JA_RE.search(text) is not None or _ZH_RE.search(text) is not None
    if lang.startswith("zh"):
        return _ZH_RE.search(text) is not None
    if lang == "ar":
        return _AR_RE.search(text) is not None

    return True

# -----------------------
# Cross-domain metaphor term detection (multilingual)
# -----------------------
def _latin_term_pattern(term: str) -> re.Pattern:
    """
    Word-boundary-ish matcher for Latin scripts to avoid false positives:
      overwhelm -> helm (should NOT match)
    Supports multiword phrases ("safe zone").
    """
    words = term.strip().split()
    escaped = [re.escape(w) for w in words if w]
    if not escaped:
        return re.compile(r"(?!x)x")  # never match
    body = r"\s+".join(escaped)
    pat = rf"(?<!\w){body}(?!\w)"
    return re.compile(pat, flags=re.IGNORECASE)

def find_cross_domain_hits(text: str, md: str, lang: str, kw_pack: dict) -> List[str]:
    forbidden_by_lang = kw_pack.get("forbidden_cross_terms_by_lang", {})
    lang_map = forbidden_by_lang.get(lang, {})
    terms: List[str] = lang_map.get(md, []) if isinstance(lang_map, dict) else []

    hits: List[str] = []
    if not terms:
        return hits

    if lang in ("en", "es", "ar"):
        for t in terms:
            if not isinstance(t, str) or not t.strip():
                continue
            pat = _latin_term_pattern(t)
            if pat.search(text):
                hits.append(t)
        return hits

    # CJK: exact substring, but skip single-character terms (noise reduction; e.g., '막').
    for t in terms:
        if not isinstance(t, str) or not t:
            continue
        if len(t) == 1:
            continue
        if t in text:
            hits.append(t)
    return hits


def find_any_domain_terms(text: str, lang: str, kw_pack: dict, max_hits: int = 12) -> List[str]:
    """
    For model_response_C (no-metaphor): detect any metaphor-domain vocabulary that leaks in.
    Uses recommended_terms_by_lang across ALL domains for the language.
    """
    rec_by_lang = kw_pack.get("recommended_terms_by_lang", {})
    lang_map = rec_by_lang.get(lang, {}) if isinstance(rec_by_lang, dict) else {}
    terms: List[str] = []
    if isinstance(lang_map, dict):
        for dom, arr in lang_map.items():
            if isinstance(arr, list):
                for t in arr:
                    if isinstance(t, str) and t:
                        terms.append(t)

    hits: List[str] = []
    if not terms:
        return hits

    if lang in ("en", "es"):
        for t in terms:
            if not isinstance(t, str) or not t.strip():
                continue
            pat = _latin_term_pattern(t)
            if pat.search(text):
                hits.append(t)
                if len(hits) >= max_hits:
                    break
        return hits

    # CJK: substring match; skip single-character terms (noise reduction).
    for t in terms:
        if not isinstance(t, str) or not t:
            continue
        if len(t) == 1:
            continue
        if t in text:
            hits.append(t)
            if len(hits) >= max_hits:
                break
    return hits


# -----------------------
# Taxonomy cache
# -----------------------
@dataclass
class TaxonomyCache:
    schema: dict
    md_tax: dict
    allowed_md: set
    md_aliases: dict
    allowed_arch: set
    subtypes_by_arch: dict
    allowed_sev: set
    allowed_fuels: set
    allowed_gv: set
    kw_pack: dict

def load_taxonomies() -> TaxonomyCache:
    schema = read_json("schema/schema_v322.json")

    md_tax = read_json("taxonomy/metaphor_domains.json")
    allowed_md = set(md_tax.get("metaphor_domains", []))
    md_aliases = md_tax.get("aliases", {})

    allowed_arch = set(read_json("taxonomy/shadow_archetypes.json").get("archetypes", []))
    subtypes_by_arch = read_json("taxonomy/shadow_subtypes.json").get("subtypes_by_archetype", {})
    allowed_sev = set(read_json("taxonomy/shadow_severities.json").get("severities", []))
    allowed_fuels = set(read_json("taxonomy/fuels.json").get("fuels", []))
    allowed_gv = set(read_json("taxonomy/gravity_vectors.json").get("gravity_vectors", []))

    kw_pack = read_json("taxonomy/metaphor_keywords.json")

    return TaxonomyCache(
        schema=schema,
        md_tax=md_tax,
        allowed_md=allowed_md,
        md_aliases=md_aliases,
        allowed_arch=allowed_arch,
        subtypes_by_arch=subtypes_by_arch,
        allowed_sev=allowed_sev,
        allowed_fuels=allowed_fuels,
        allowed_gv=allowed_gv,
        kw_pack=kw_pack,
    )

def norm_metaphor_domain(md: Any, aliases: dict) -> Any:
    if not isinstance(md, str) or not md:
        return md
    return aliases.get(md, md)

# -----------------------
# Validation
# -----------------------
def validate_output(out_obj: dict, expected_lang: str, tax: TaxonomyCache) -> Tuple[str, List[str]]:
    # schema가 qa_ok / qa_fail_reasons 를 required로 잡고 있으면,
    # 검증 전에 placeholder를 넣어 'missing required field'를 막는다.
    out_obj.setdefault("qa_ok", "YES")
    out_obj.setdefault("qa_fail_reasons", [])

    
    fail_reasons: List[str] = []
    # Required fields for A/B/C dataset (make schema-valid, but still fail QA to trigger retry)
    if ("diff_notes" not in out_obj) or (not isinstance(out_obj.get("diff_notes"), dict)):
        fail_reasons.append("SCHEMA: missing or invalid field 'diff_notes'")
        out_obj["diff_notes"] = {"A_intent": "", "B_edit_summary": "", "C_edit_summary": ""}
    else:
        dn = out_obj["diff_notes"]
        dn.setdefault("A_intent", "")
        dn.setdefault("B_edit_summary", "")
        dn.setdefault("C_edit_summary", "")

    if ("model_response_C" not in out_obj) or (not isinstance(out_obj.get("model_response_C"), dict)):
        fail_reasons.append("SCHEMA: missing or invalid field 'model_response_C'")
        out_obj["model_response_C"] = {
            "reflection": "",
            "anchor": "",
            "route_steps": [],
            "micro_script": "",
            "one_question": "",
            "closing": ""
        }
    else:
        c = out_obj["model_response_C"]
        for k in ("reflection", "anchor", "route_steps", "micro_script", "one_question", "closing"):
            if k not in c:
                c[k] = [] if k == "route_steps" else ""


    qa = out_obj.get("qa")
    if not isinstance(qa, dict):
        qa = {}
        out_obj["qa"] = qa

    # normalize qa.metaphor_domain via aliases
    md = qa.get("metaphor_domain")
    if isinstance(md, str):
        md_norm = norm_metaphor_domain(md, tax.md_aliases)
        if md_norm != md:
            qa["metaphor_domain"] = md_norm
            md = md_norm

        if md_norm and md_norm not in tax.allowed_md:
            fail_reasons.append(f"TAXONOMY: qa.metaphor_domain '{md_norm}' not in allowed list")
    elif md is not None:
        fail_reasons.append("TAXONOMY: qa.metaphor_domain must be a string")

    # consistency: qa.metaphor_domain == A.aroma.metaphor_domain
    try:
        a = out_obj.get("model_response_A", {})
        if isinstance(a, dict):
            aroma = a.get("aroma")
            if isinstance(aroma, dict):
                md2 = aroma.get("metaphor_domain")
                if isinstance(md2, str):
                    md2_norm = norm_metaphor_domain(md2, tax.md_aliases)
                    if md2_norm != md2:
                        aroma["metaphor_domain"] = md2_norm
                        md2 = md2_norm
                else:
                    md2 = None

                if isinstance(md, str) and isinstance(md2, str) and md != md2:
                    fail_reasons.append(
                        f"CONSISTENCY: qa.metaphor_domain '{md}' != model_response_A.aroma.metaphor_domain '{md2}'"
                    )
    except Exception as e:
        fail_reasons.append(f"CONSISTENCY_CHECK_ERROR: {str(e)}")

    # schema validate
    if "qa_ok" not in out_obj:
        fail_reasons.append("SCHEMA: missing required field 'qa_ok'")
        out_obj["qa_ok"] = "NO"
    if "qa_fail_reasons" not in out_obj:
        fail_reasons.append("SCHEMA: missing required field 'qa_fail_reasons'")
        out_obj["qa_fail_reasons"] = []

    try:
        validate(instance=out_obj, schema=tax.schema)
    except ValidationError as e:
        path = ".".join([str(p) for p in e.absolute_path])
        fail_reasons.append(f"SCHEMA[{path}]: {e.message}")
    except Exception as e:
        fail_reasons.append(f"SCHEMA_CHECK_ERROR: {str(e)}")

    # shadow taxonomy
    signals = out_obj.get("signals", {})
    if not isinstance(signals, dict):
        signals = {}

    shadow = signals.get("shadow", {})
    if not isinstance(shadow, dict):
        shadow = {}

    arch = shadow.get("archetype")
    subtype = shadow.get("subtype")
    sev = shadow.get("severity")

    if isinstance(arch, str) and arch and arch not in tax.allowed_arch:
        fail_reasons.append(f"TAXONOMY: shadow.archetype '{arch}' not allowed")
    if isinstance(sev, str) and sev and sev not in tax.allowed_sev:
        fail_reasons.append(f"TAXONOMY: shadow.severity '{sev}' not allowed")

    if isinstance(arch, str) and isinstance(subtype, str) and arch and subtype:
        allowed_subtypes = set(tax.subtypes_by_arch.get(arch, []))
        if allowed_subtypes and subtype not in allowed_subtypes:
            fail_reasons.append(f"TAXONOMY: shadow.subtype '{subtype}' not valid for archetype '{arch}'")

    # fuels
    fuel = signals.get("fuel", {})
    if not isinstance(fuel, dict):
        fuel = {}
    for k in ("primary", "secondary"):
        v = fuel.get(k)
        if isinstance(v, str) and v and v not in tax.allowed_fuels:
            fail_reasons.append(f"TAXONOMY: fuel.{k} '{v}' not allowed")

    # gravity_vectors
    gvs = signals.get("gravity_vectors", [])
    if gvs is None:
        gvs = []
    if isinstance(gvs, list):
        for gv in gvs:
            if isinstance(gv, str) and gv and gv not in tax.allowed_gv:
                fail_reasons.append(f"TAXONOMY: gravity_vector '{gv}' not allowed")
            elif not isinstance(gv, str):
                fail_reasons.append("TAXONOMY: gravity_vectors must contain strings only")
    else:
        fail_reasons.append("TAXONOMY: signals.gravity_vectors must be an array of strings")

    # language (hard fail)
    blob = collect_text(out_obj)
    if expected_lang and not validate_language(blob, expected_lang):
        fail_reasons.append(f"LANGUAGE: output does not look like expected language '{expected_lang}'")

    # STYLE WARNING: cross-domain terms (not fail)
    try:
        md_now = qa.get("metaphor_domain")
        lang_now = qa.get("language") or expected_lang or "en"
        hits = find_cross_domain_hits(blob, md_now, lang_now, tax.kw_pack) if isinstance(md_now, str) else []
        if hits:
            qa.setdefault("style_warnings", [])
            qa["style_warnings"].append(f"Cross-domain terms for '{md_now}' ({lang_now}): {hits}")
        # model_response_C should be NO-metaphor: warn if metaphor markers or domain vocab appears
        try:
            c = out_obj.get("model_response_C", {})
            if isinstance(c, dict):
                c_parts: List[str] = []
                for k in ("reflection", "anchor", "micro_script", "closing", "one_question"):
                    c_parts.append(str(c.get(k, "")))
                rs_c = c.get("route_steps", [])
                if isinstance(rs_c, list):
                    c_parts.extend([str(x) for x in rs_c])
                c_blob = "\n".join([p for p in c_parts if p])

                # For the CS dataset, "plain" C may still use normal business words.
                # We only warn on obvious metaphor markers and a small set of imagery-heavy tokens.
                profile = (os.getenv("DATASET_PROFILE", "noble") or "noble").strip().lower()

                # simple metaphor/simile markers
                markers: List[str] = []
                low = c_blob.lower()
                for s in ("like a", "as if", "as though"):
                    if s in low:
                        markers.append(s)
                if lang_now == "ko":
                    for s in ("처럼", "마치"):
                        if s in c_blob:
                            markers.append(s)

                imagery_terms: List[str] = []
                hits_any: List[str] = []
                if profile != "cs":
                    # Non-CS: keep legacy strictness (domain vocabulary leakage)
                    hits_any = find_any_domain_terms(c_blob, lang_now, tax.kw_pack)
                else:
                    # CS: allow "breath" and other neutral grounding; warn on common metaphor nouns.
                    banned = [
                        "wave", "tide", "storm", "ocean", "underwater", "anchor", "compass", "route", "map",
                        "trail", "peak", "cliff", "garden", "seed", "root", "stage", "spotlight",
                        "track", "tempo", "refrain", "sheet music", "deck", "ship",
                    ]
                    for tkn in banned:
                        if _latin_term_pattern(tkn).search(c_blob):
                            imagery_terms.append(tkn)

                if markers or hits_any or imagery_terms:
                    qa.setdefault("style_warnings", [])
                    msg = "C contains metaphor markers/terms: "
                    bits = []
                    if markers:
                        bits.append(f"markers={markers[:8]}")
                    if imagery_terms:
                        bits.append(f"terms={imagery_terms[:8]}")
                    elif hits_any:
                        bits.append(f"terms={hits_any[:8]}")
                    qa["style_warnings"].append(msg + "; ".join(bits))
        except Exception:
            pass

    except Exception:
        pass

    return ("NO", fail_reasons) if fail_reasons else ("YES", [])

# -----------------------
# Model calls
# -----------------------
def try_generate_once(model: str, instructions: str, input_text: str, effort: str) -> str:
    resp = client.responses.create(
        model=model,
        instructions=instructions,
        input=input_text,
        reasoning={"effort": effort},
    )
    return (resp.output_text or "").strip()

def build_quality_repair_instructions(
    base_instructions: str,
    expected_lang: str,
    md: str,
    qa_fail: List[str],
    style_warnings: List[str],
) -> str:
    # Extract concrete banned terms from style warnings so the retry can reliably remove them.
    banned_terms: List[str] = []
    try:
        import ast

        for w in (style_warnings or []):
            if not isinstance(w, str):
                continue

            if w.startswith("Cross-domain terms") and ":" in w:
                tail = w.split(":", 1)[1].strip()
                try:
                    vals = ast.literal_eval(tail)
                    if isinstance(vals, list):
                        banned_terms.extend([str(x) for x in vals if x])
                except Exception:
                    pass

            if "terms=" in w:
                try:
                    tail = w.split("terms=", 1)[1].strip()
                    # tail might look like "['support']" or "['a','b']; markers=[...]"
                    tail = tail.split(";", 1)[0].strip()
                    vals = ast.literal_eval(tail)
                    if isinstance(vals, list):
                        banned_terms.extend([str(x) for x in vals if x])
                except Exception:
                    pass
    except Exception:
        banned_terms = []

    # De-dup while preserving order
    seen = set()
    banned_terms = [t for t in banned_terms if not (t in seen or seen.add(t))]

    repair_lines = [
        "",
        "# REPAIR MODE (one retry only)",
        f"- Output language MUST be '{expected_lang}'.",
        f"- For model_response_A and model_response_B: keep metaphor domain STRICTLY as '{md}'. Do not mix domains.",
        "- For model_response_C: NO metaphors / NO imagery / NO domain vocabulary. Use plain modern conversational tone.",
        "- Ensure required fields exist: model_response_A, model_response_B, model_response_C, diff_notes, qa_ok, qa_fail_reasons.",
        "- Ensure schema types: route_steps is an array of strings; anchor is a string; diff_notes is an object.",
        "- Output ONLY valid JSON (double quotes). No markdown. No extra text.",
    ]
    if banned_terms:
        repair_lines.append(
            "- Remove these exact terms/phrases that triggered style_warnings; replace with neutral alternatives: "
            + ", ".join(banned_terms[:20])
        )
    if qa_fail:
        repair_lines.append(f"- Previous QA failures (first 8): {qa_fail[:8]}")
    if style_warnings:
        repair_lines.append(f"- Previous style_warnings (first 8): {style_warnings[:8]}")
    return base_instructions + "\n" + "\n".join(repair_lines)


def build_json_repair_instructions(expected_lang: str, md: str) -> str:
    return "\n".join([
        "You are a strict JSON repair bot.",
        f"- Output language MUST be '{expected_lang}'.",
        f"- Keep metaphor domain STRICTLY as '{md}'.",
        "- Convert the given text into ONE valid JSON object.",
        "- Preserve all keys/values you can; do not invent new facts.",
        "- Ensure all strings use double quotes, arrays/objects are valid, and commas are correct.",
        "- Output ONLY the fixed JSON. No markdown. No commentary.",
    ])

# -----------------------
# Main
# -----------------------
def main() -> None:
    tax = load_taxonomies()
    template = read_json(TEMPLATE_PATH)
    scenarios = read_scenarios_txt(SCENARIOS_PATH)
    # Auto-select prompt profile if user didn't set it.
    if DATASET_PROFILE == "cs" and not os.getenv("PROMPT_PROFILE"):
        os.environ["PROMPT_PROFILE"] = "cs"
    base_instructions = build_instructions()

    print(f"템플릿: {TEMPLATE_PATH}")
    print(f"시나리오 수: {len(scenarios)}")
    print(f"출력: {OUT_PATH}")
    print(f"도메인 강제순환(FORCE_ROTATE_DOMAINS): {FORCE_ROTATE_DOMAINS}")
    print(f"추론 강도(REASONING_EFFORT): {REASONING_EFFORT}")
    print(f"재시도(MAX_RETRY): {MAX_RETRY} | 스타일경고 재시도(RETRY_ON_STYLE_WARNINGS): {RETRY_ON_STYLE_WARNINGS}")
    print(f"데이터셋 프로필(DATASET_PROFILE): {DATASET_PROFILE}")

    domain_i = 0

    for i, raw_line in enumerate(scenarios, start=1):
        prefix_md, scenario_text = parse_scenario_prefix(raw_line)

        scenario_obj = deepcopy(template)
        scenario_obj["scenario"] = scenario_text

        # CS profile: overwrite therapy defaults with CS presets
        if DATASET_PROFILE == "cs":
            apply_cs_presets(scenario_obj, scenario_text)

        # Decide domain
        # - CS profile: default to a single, professional metaphor domain (ARCH) to avoid weird variety.
        # - Prefix "ARCH|..." still works.
        if DATASET_PROFILE == "cs":
            if isinstance(prefix_md, str):
                scenario_obj.setdefault("qa", {})["metaphor_domain"] = prefix_md
            else:
                scenario_obj.setdefault("qa", {})["metaphor_domain"] = "ARCH"
        else:
            if FORCE_ROTATE_DOMAINS:
                scenario_obj.setdefault("qa", {})["metaphor_domain"] = ALLOWED_DOMAINS[domain_i % len(ALLOWED_DOMAINS)]
                domain_i += 1
            elif isinstance(prefix_md, str):
                scenario_obj.setdefault("qa", {})["metaphor_domain"] = prefix_md
            else:
                md = scenario_obj.get("qa", {}).get("metaphor_domain")
                if not isinstance(md, str) or not md.strip():
                    scenario_obj.setdefault("qa", {})["metaphor_domain"] = ALLOWED_DOMAINS[domain_i % len(ALLOWED_DOMAINS)]
                    domain_i += 1

        # Inject qa.language + qa.metaphor_pack
        lang = detect_language(scenario_text)
        inject_qa_defaults(scenario_obj, language=lang, kw_pack_version=tax.kw_pack.get("version"))

        input_text = json.dumps(scenario_obj, ensure_ascii=False)

        # 0) First generation
        raw = try_generate_once(MODEL, base_instructions, input_text, effort=REASONING_EFFORT)
        out_obj = try_parse_json(raw)

        # 0b) JSON repair (one shot)
        if out_obj is None:
            save_bad_output(i, raw, suffix="_gen")
            md_hint = scenario_obj.get("qa", {}).get("metaphor_domain") or "NAV"
            repair_inst = build_json_repair_instructions(expected_lang=lang, md=md_hint)
            raw_fix = try_generate_once(MODEL, repair_inst, raw, effort=REASONING_EFFORT)
            out_obj = try_parse_json(raw_fix)
            if out_obj is None:
                save_bad_output(i, raw_fix, suffix="_json_repair")
                raise RuntimeError(
                    f"[{i}] JSON 파싱 실패(수리 1회 포함). bad_outputs/bad_output_{i:05d}_gen.txt / _json_repair.txt 확인!"
                )
        
        # ✅ (여기!) 원본 scenario_obj을 베이스로 깔고, 모델 출력은 "추가 필드만" 얹기
        out_obj, override_notes = merge_additions_preserve_original(scenario_obj, out_obj)
        if override_notes:
            out_obj.setdefault("qa", {})
            out_obj["qa"].setdefault("style_warnings", [])
            out_obj["qa"]["style_warnings"].append(
                f"Preserve-original overrides ignored: {override_notes[:10]}"
            )

        # 1) Postprocess + QA
        coerce_house_style_strings(out_obj)
        qa_ok, qa_fail = validate_output(out_obj, expected_lang=lang, tax=tax)

        out_obj["qa_ok"] = qa_ok
        out_obj["qa_fail_reasons"] = qa_fail

        if "id" not in out_obj:
            out_obj["id"] = datetime.now(timezone.utc).strftime("sample_%Y%m%d_%H%M%S_%f")

        # 2) Optional quality retry
        attempts = 0
        while attempts < MAX_RETRY:
            style_warnings = out_obj.get("qa", {}).get("style_warnings", [])
            need_retry = (qa_ok != "YES") or (RETRY_ON_STYLE_WARNINGS and len(style_warnings) > 0)
            if not need_retry:
                break

            md_now = out_obj.get("qa", {}).get("metaphor_domain") or scenario_obj.get("qa", {}).get("metaphor_domain") or "NAV"
            repair_inst = build_quality_repair_instructions(
                base_instructions=base_instructions,
                expected_lang=lang,
                md=md_now,
                qa_fail=qa_fail,
                style_warnings=style_warnings,
            )

            raw2 = try_generate_once(MODEL, repair_inst, input_text, effort=REASONING_EFFORT)
            out_obj2 = try_parse_json(raw2)
            if out_obj2 is None:
                save_bad_output(i, raw2, suffix=f"_retry{attempts+1}_badjson")
                attempts += 1
                continue

            coerce_house_style_strings(out_obj2)
            qa_ok2, qa_fail2 = validate_output(out_obj2, expected_lang=lang, tax=tax)
            out_obj2["qa_ok"] = qa_ok2
            out_obj2["qa_fail_reasons"] = qa_fail2
            if "id" not in out_obj2:
                out_obj2["id"] = out_obj["id"]

            # adopt rule
            sw2 = out_obj2.get("qa", {}).get("style_warnings", [])
            sw1 = out_obj.get("qa", {}).get("style_warnings", [])

            if qa_ok2 == "YES" and len(sw2) == 0:
                out_obj, qa_ok, qa_fail = out_obj2, qa_ok2, qa_fail2
            elif qa_ok2 == "YES" and qa_ok != "YES":
                out_obj, qa_ok, qa_fail = out_obj2, qa_ok2, qa_fail2
            elif qa_ok2 == qa_ok and len(sw2) < len(sw1):
                out_obj, qa_ok, qa_fail = out_obj2, qa_ok2, qa_fail2

            attempts += 1

        # 3) Save + log (한 번만!)
        append_jsonl(OUT_PATH, out_obj)

        md_print = out_obj.get("qa", {}).get("metaphor_domain")
        status = "✅" if out_obj.get("qa_ok") == "YES" else "⚠️"
        print(f"{status} [{i}/{len(scenarios)}] 저장됨 id={out_obj['id']} md={md_print} lang={lang} qa_ok={out_obj['qa_ok']}")

        if out_obj.get("qa_fail_reasons"):
            for r in out_obj["qa_fail_reasons"]:
                print("   -", r)

        warnings = out_obj.get("qa", {}).get("style_warnings", [])
        if warnings:
            print("⚠️ STYLE WARNINGS:")
            for w in warnings:
                print("   -", w)

if __name__ == "__main__":
    main()