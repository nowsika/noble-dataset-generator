import json
import os
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
from openai import OpenAI

from jsonschema import validate
from jsonschema.exceptions import ValidationError

#프롬프트 편지3종 합치기#
from prompt_builder import build_instructions
from auto_inject import inject_qa_defaults

load_dotenv()
client = OpenAI()

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")

# Prompt letter variant(s): A/B/C. Default A.
# Examples: PROMPT_SET="C" or PROMPT_SET="ABC" or PROMPT_SET="A,B,C"
PROMPT_SET = os.getenv("PROMPT_SET", "A")

def read_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def append_jsonl(path: str, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def norm_metaphor_domain(md: str, aliases: dict) -> str:
    if not md:
        return md
    return aliases.get(md, md)

def main():
    in_path = os.getenv("SCENARIO_PATH", "scenario.json")
    out_path = os.getenv("OUTPUT_PATH", "outputs.jsonl")

    scenario_obj = read_json(in_path)

    # Auto-inject qa.language + qa.metaphor_pack using "언어=질문 언어" rule
    scenario_obj = inject_qa_defaults(scenario_obj)

    # Parse variants
    raw_sets = (PROMPT_SET or "A").strip()
    if "," in raw_sets:
        candidates = [v.strip().upper() for v in raw_sets.split(",") if v.strip()]
    else:
        candidates = [c.upper() for c in raw_sets if c.strip()]
    variants = []
    for v in candidates:
        if v in ("A", "B", "C") and v not in variants:
            variants.append(v)
    if not variants:
        variants = ["A"]

    # Load schema + taxonomies once
    schema = read_json("schema/schema_v322.json")

    md_tax = read_json("taxonomy/metaphor_domains.json")
    allowed_md = set(md_tax.get("metaphor_domains", []))
    md_aliases = md_tax.get("aliases", {})

    allowed_arch = set(read_json("taxonomy/shadow_archetypes.json")["archetypes"])
    subtypes_by_arch = read_json("taxonomy/shadow_subtypes.json")["subtypes_by_archetype"]
    allowed_sev = set(read_json("taxonomy/shadow_severities.json")["severities"])
    allowed_fuels = set(read_json("taxonomy/fuels.json")["fuels"])
    allowed_gv = set(read_json("taxonomy/gravity_vectors.json")["gravity_vectors"])

    for v in variants:
        scenario_run = json.loads(json.dumps(scenario_obj, ensure_ascii=False))  # deep-ish copy
        scenario_run.setdefault("qa", {})["prompt_set"] = v
        input_text = json.dumps(scenario_run, ensure_ascii=False)

        instructions = build_instructions(letter_variant=v)

        resp = client.responses.create(
            model=MODEL,
            instructions=instructions,
            input=input_text,
        )

        raw = (resp.output_text or "").strip()

        try:
            out_obj = json.loads(raw)
        except json.JSONDecodeError:
            Path(f"bad_output_{v}.txt").write_text(raw, encoding="utf-8")
            raise RuntimeError(f"[{v}] JSON 파싱 실패. bad_output_{v}.txt 확인해줘!")

        # --- Schema + taxonomy validation ---
        fail_reasons = []

        try:
            # placeholders for required fields (schema-valid), but keep fail reasons if missing
            if ("diff_notes" not in out_obj) or (not isinstance(out_obj.get("diff_notes"), dict)):
                fail_reasons.append("SCHEMA: missing or invalid field 'diff_notes'")
                out_obj["diff_notes"] = {"A_intent": "", "B_edit_summary": "", "C_edit_summary": ""}
            else:
                out_obj["diff_notes"].setdefault("A_intent", "")
                out_obj["diff_notes"].setdefault("B_edit_summary", "")
                out_obj["diff_notes"].setdefault("C_edit_summary", "")

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


            validate(instance=out_obj, schema=schema)
        except ValidationError as e:
            fail_reasons.append(f"SCHEMA: {e.message}")

        md = out_obj.get("qa", {}).get("metaphor_domain")
        md_norm = norm_metaphor_domain(md, md_aliases)
        if md_norm and md_norm not in allowed_md:
            fail_reasons.append(f"TAXONOMY: qa.metaphor_domain '{md}' not in allowed list")
        else:
            if md and md_norm != md:
                out_obj.setdefault("qa", {})["metaphor_domain"] = md_norm

        shadow = out_obj.get("signals", {}).get("shadow", {}) or {}
        arch = shadow.get("archetype")
        subtype = shadow.get("subtype")
        sev = shadow.get("severity")

        if arch and arch not in allowed_arch:
            fail_reasons.append(f"TAXONOMY: shadow.archetype '{arch}' not allowed")
        if sev and sev not in allowed_sev:
            fail_reasons.append(f"TAXONOMY: shadow.severity '{sev}' not allowed")

        if arch and subtype:
            allowed_subtypes = set(subtypes_by_arch.get(arch, []))
            if subtype not in allowed_subtypes:
                fail_reasons.append(
                    f"TAXONOMY: shadow.subtype '{subtype}' not valid for archetype '{arch}'"
                )

        fuel = out_obj.get("signals", {}).get("fuel", {}) or {}
        for k in ("primary", "secondary"):
            vv = fuel.get(k)
            if vv and vv not in allowed_fuels:
                fail_reasons.append(f"TAXONOMY: fuel.{k} '{vv}' not allowed")

        gvs = out_obj.get("signals", {}).get("gravity_vectors", []) or []
        if isinstance(gvs, list):
            for gv in gvs:
                if isinstance(gv, str) and gv not in allowed_gv:
                    fail_reasons.append(f"TAXONOMY: gravity_vector '{gv}' not allowed")
        else:
            fail_reasons.append("TAXONOMY: signals.gravity_vectors must be an array of strings")

        out_obj["qa_ok"] = "NO" if fail_reasons else "YES"
        out_obj["qa_fail_reasons"] = fail_reasons

        if "id" not in out_obj:
            out_obj["id"] = datetime.now(timezone.utc).strftime(f"sample_%Y%m%d_%H%M%S_%f_{v}")

        append_jsonl(out_path, out_obj)

        status = "✅" if out_obj.get("qa_ok") == "YES" else "⚠️"
        print(f"{status} 저장 완료: {out_path}")
        print(f"   set: {v}")
        print(f"   id: {out_obj.get('id')}")
        if fail_reasons:
            print("⚠️ QA FAIL:")
            for r in fail_reasons:
                print(" -", r)

if __name__ == "__main__":
    main()
