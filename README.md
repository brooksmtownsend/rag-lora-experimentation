# LLM Fine-Tuning + RAG Demo (DMN JSON for Banking)

This repo is a compact demo that compares:

- Base model vs. RAG vs. LoRA vs. RAG+LoRA
- Banking policy recall (RAG helps)
- DMN JSON formatting consistency (LoRA helps)

It is intentionally small and fast so you can run it live.

## Quick Start

1. Create/activate a virtualenv and install deps.
2. Generate data.
3. Train a LoRA adapter.
4. Run the comparison script.

### 1) Install dependencies

Use your existing venv or create one:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the core packages:

```bash
pip install torch transformers datasets peft sentence-transformers faiss-cpu
```

### 2) Generate data

```bash
python3 make_data.py
```

### 3) Train LoRA

```bash
python3 train_lora.py
```

The default training set (`data/lora_train.jsonl`) is now DMN JSON.

### 4) Run the demo comparison

```bash
python3 run_compare.py
```

## Demo Script (Live Walkthrough)

Use this script when showing improvements live:

1. Regenerate data to show the dataset is simple + editable:
   `python3 make_data.py`
2. Train the LoRA adapter:
   `python3 train_lora.py`
3. Run the comparison to show 4 modes:
   `python3 run_compare.py`
4. Call out:
   - RAG improves policy recall
   - LoRA improves DMN JSON format consistency
   - RAG+LoRA combines both
   - DMN JSON output is validated and auto-repaired if invalid

## Configuration

Override the base model:

```bash
MODEL_ID=TinyLlama/TinyLlama-1.1B-Chat-v1.0 python3 train_lora.py
MODEL_ID=TinyLlama/TinyLlama-1.1B-Chat-v1.0 python3 run_compare.py
```

Override the training file:

```bash
TRAIN_FILE=data/lora_train.jsonl python3 train_lora.py
```

## DMN JSON Validation

The demo includes a simple DMN JSON validator with a retry loop. If the model
emits invalid JSON or misses required DMN fields, the script re-prompts it with
the validation errors and requests corrected JSON only.

You can also tune demo speed vs. completeness:

```bash
DMN_MAX_NEW_TOKENS=400 DMN_MAX_RETRIES=1 python3 run_compare.py
```

Lower `DMN_MAX_NEW_TOKENS` to speed up CPU demos at the cost of truncation risk.

## Offline / Cached Mode

If you already have models cached in `~/.cache/huggingface`, you can run offline:

```bash
HF_HOME=$HOME/.cache/huggingface \
HF_DATASETS_CACHE=data/hf_datasets \
TRANSFORMERS_OFFLINE=1 \
HF_HUB_OFFLINE=1 \
python3 train_lora.py
```

Same for the demo:

```bash
HF_HOME=$HOME/.cache/huggingface \
TRANSFORMERS_OFFLINE=1 \
HF_HUB_OFFLINE=1 \
python3 run_compare.py
```

If you update `data/rag_corpus.jsonl`, delete `data/rag_cache` to rebuild embeddings.

## Appendix: How To Provide More Accurate Data

The model output quality is dominated by the example quality. Here’s a quick playbook.

### 1) Separate stable rules from fresh facts

- **Stable conventions** (format, schema, error handling) go into LoRA training examples.
- **Fresh facts** (policy updates, product changes, current examples) go into the RAG corpus.

### 2) Write “gold” examples for each intent

For each business intent, include:

- A clean, canonical input prompt
- A fully-correct output that matches your DMN JSON schema
- At least one negative example showing what “unsupported” looks like

### 3) Make outputs unambiguous

Avoid optional fields or vague wording in your outputs.
Prefer explicit keys and consistent ordering. For DMN JSON:

- Always include required keys
- Use canonical values (e.g., `status = "error"` not `status = "ERR"`)
- Keep strict formatting with no commentary

### 4) Add edge cases early

Business users will write messy input. Cover:

- Missing required fields
- Conflicting constraints
- Unsupported operations
- Partial inputs that need validation errors

### 5) Prefer small, targeted updates

It’s better to add 5 high-quality examples that cover a new rule
than 50 low-quality or redundant examples.

### 6) Keep RAG entries short and tagged

Chunk your RAG docs (200–800 chars) and include:

- `id` and `title`
- Clear policy details and exact DMN JSON examples
- The current recommended pattern (not the old one)

---

If you want, I can add a JSON schema validator + retry loop for demos,
or expand the datasets to include more realistic business process prompts.
