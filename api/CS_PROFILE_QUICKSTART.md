# CS profile quickstart

This repo ships with two dataset "profiles":

- **noble** (default): counseling-flavored prompts and therapy-like default tags.
- **cs**: professional B2B SaaS customer support dataset.

## Generate CS dataset

1) Put your scenarios in a text file (one scenario per line), e.g. `scenarios_300.txt`.
2) Set env vars and run:

```powershell
$env:DATASET_PROFILE="cs"
$env:PROMPT_PROFILE="cs"
$env:TEMPLATE_PATH="scenario_template_cs.json"
$env:SCENARIOS_PATH="scenarios_300.txt"
$env:OUTPUT_PATH="outputs.jsonl"
# optional: keep metaphor domain fixed (CS profile ignores rotation and uses ARCH unless you prefix a domain)
$env:FORCE_ROTATE_DOMAINS="false"
python batch_generate.py
```

## What changes in CS profile

- Overwrites `signals`, `tags`, and `context_state.tone_hint` per scenario using keyword-based categorization.
- Fixes `qa.metaphor_domain` to `ARCH` by default to keep metaphors minimal and professional.
- Uses `prompts/system_core_cs.txt` and skips the counseling letters.
- Makes C-style warnings less trigger-happy (no more "breath" false positives).
