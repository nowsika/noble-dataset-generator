````md
# noble-dataset-generator

Generate JSONL datasets from the NOBLE Engine 3.0 alignment spec (OpenAI API pipeline).

## Quickstart

```bash
git clone <this-repo>
cd noble-dataset-generator

python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
````

### Configure

1. Copy `.env.example` → `.env`
2. Put your OpenAI key into `.env`

> Note: `.env` is ignored by git. Never commit your real API key.

### Run

```bash
python api/batch_generate.py
```

## How it works (inputs → output)

### Input

Write scenarios in:

* `api/scenarios.txt` (one scenario per line)

Example:

```txt
I got invited but the link expired - can you resend a fresh invite?
Our invoice total doesn't match the seat count shown in the admin console.
We suspect a compromised account—what security steps should we take?
```

Templates:

* `api/scenario_template.json` (base template)
* `api/scenario_template_cs.json` (CS-specific template, if used by presets)

### Output

The generated dataset is written as JSONL (one JSON object per line).

* Output file: `api/outputs.jsonl`

## Notes

* Recommended: keep `api/outputs.jsonl` git-ignored (it changes on every run).
* If you share samples, put them in `data/` (e.g. `data/noble_dataset_sample.jsonl`).
