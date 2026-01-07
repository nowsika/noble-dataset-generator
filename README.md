````md
# noble-dataset-generator
Generate JSONL datasets from the NOBLE Engine 3.0 alignment spec (OpenAI API pipeline).

> **Project NOBLE** is a framework designed to inject 'Nobility' and 'Internal Conscience' into AI models through a specialized alignment layer. This generator helps you create high-quality synthetic datasets based on that philosophy.

## Quickstart

### 1) Install
```bash
git clone https://github.com/nowsika/noble-dataset-generator.git
cd noble-dataset-generator

python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
````

### 2) Configure

1. Copy `.env.example` → `.env`
2. Put your OpenAI API key into `.env`

> `.env` is git-ignored. Never commit your real API key.

### 3) Write scenarios

Add one scenario per line:

* `api/scenarios.txt`

Example:

```
I got invited but the link expired - can you resend a fresh invite?
Our invoice total doesn't match the seat count shown in the admin console.
We suspect a compromised account—what security steps should we take?
```

### 4) Run

```bash
python api/batch_generate.py
```

## Output

* Output file: `api/outputs.jsonl` (JSONL: one JSON object per line)

## Templates

* `api/scenario_template.json` (base template)
* `api/scenario_template_cs.json` (CS preset template)

## Notes

* Keep `api/outputs.jsonl` git-ignored (it changes every run).
* If you want to share samples, put them in `data/` (e.g. `data/noble_dataset_sample.jsonl`).

## Attribution
If you use this work, please cite/credit:
- **NOBLE Engine 3.0 — by Young-Hun Choe**
