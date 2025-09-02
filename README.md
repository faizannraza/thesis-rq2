# RQ2 — 30-Day Freshness on GPU (H100) with GPT-OSS-20B

**Branch:** `rq2-GPU`  
**Model:** `openai/gpt-oss-20b` for everything (baseline + LoRA/Replay/Hybrid)  
**Goal:** Compare three continual-learning regimens over 30 synthetic “days” of new facts:

- **LoRA-only** (cheap daily adapters)  
- **Replay-only** (memory buffer)  
- **Hybrid** (LoRA + Replay 1:1)  

Each “day” we compute:

- **FFI (Freshness):** exact-match on day-new questions  
- **Legacy Accuracy:** exact-match on a fixed, pre-update question set  

> ⚡ This branch is tuned for a single NVIDIA H100 (80GB recommended).  
> If you’re on a smaller GPU, see the **Troubleshooting / Memory knobs** section.

---

## 1. Quick Start (H100)

```bash
# 0) Get the code (this assumes your repo folder is ~/thesis-rq2)
REPO_DIR=~/thesis-rq2
cd "$REPO_DIR"
git fetch origin
git checkout rq2-GPU
git pull --ff-only

# 1) Fresh venv + deps
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# If your torch wheel isn't the CUDA build, install the official CUDA wheel:
# pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio

# Optional: faster checkpoints from HF Hub
export HF_HUB_ENABLE_HF_TRANSFER=1

# 2) Sanity: print device info (harmless without nvidia-smi)
python -c 'import torch,platform,subprocess,shutil; print("Torch", torch.__version__); print("CUDA:", torch.cuda.is_available()); print("Device:", "cuda" if torch.cuda.is_available() else "cpu"); print(subprocess.run(["nvidia-smi","--query-gpu=name,memory.total,driver_version","--format=csv,noheader"], capture_output=True, text=True).stdout if shutil.which("nvidia-smi") else "(no nvidia-smi)")'

# 3) Build the synthetic “30-day” dataset (Wikitext slices + cloze QA)
python scripts/prepare_days.py --days_dir data/days --legacy_dir data/legacy --days 30

# 4) Use GPT-OSS-20B for baseline + regimens
export MODEL="openai/gpt-oss-20b"

# 5) Baseline (no updates) on Day-1
python scripts/eval_qa.py \
  --model_name "$MODEL" \
  --qa_file data/days/qa_eval_day_01.jsonl \
  --out_dir logs/smoke_baseline_oss20b \
  --device auto \
  --max_eval 150 \
  --max_new_tokens 16

# 6) 3-day smoke runs (LoRA / Replay / Hybrid) — ALL on GPT-OSS-20B
# Conservative settings to avoid OOM on 20B (short context, tiny batch, grad-accum).

python scripts/day_loop.py --base_model "$MODEL" \
  --regimen lora_only --max_days 3 \
  --out_root artifacts/rq2_lora_only_smoke_oss20b \
  --adapters_root models/adapters \
  --days_dir data/days --legacy_eval data/legacy/qa_legacy_holdout.jsonl \
  --epochs_per_day 1 --bsz 1 --grad_accum 4 --max_len 256 \
  --max_eval 150 --device auto

python scripts/day_loop.py --base_model "$MODEL" \
  --regimen replay_only --max_days 3 \
  --out_root artifacts/rq2_replay_only_smoke_oss20b \
  --adapters_root models/adapters \
  --days_dir data/days --legacy_eval data/legacy/qa_legacy_holdout.jsonl \
  --epochs_per_day 1 --bsz 1 --grad_accum 4 --max_len 256 \
  --max_eval 150 --device auto

python scripts/day_loop.py --base_model "$MODEL" \
  --regimen hybrid --max_days 3 \
  --out_root artifacts/rq2_hybrid_smoke_oss20b \
  --adapters_root models/adapters \
  --days_dir data/days --legacy_eval data/legacy/qa_legacy_holdout.jsonl \
  --epochs_per_day 1 --bsz 1 --grad_accum 4 --max_len 256 \
  --max_eval 150 --device auto

# 7) Plot + summarize (figures + final day CSV)
python scripts/analyze_rq2.py

# 8) Bundle deliverables (no large data/adapter dirs)
mkdir -p deliverables
cp -f artifacts/figures/ffi_over_days.png artifacts/figures/legacy_over_days.png artifacts/figures/final_day_summary.csv deliverables/ 2>/dev/null || true
cp -f artifacts/rq2_lora_only_smoke_oss20b/summary.csv    deliverables/summary_lora_only_smoke_oss20b.csv 2>/dev/null || true
cp -f artifacts/rq2_replay_only_smoke_oss20b/summary.csv  deliverables/summary_replay_only_smoke_oss20b.csv 2>/dev/null || true
cp -f artifacts/rq2_hybrid_smoke_oss20b/summary.csv       deliverables/summary_hybrid_smoke_oss20b.csv 2>/dev/null || true
cp -f logs/smoke_baseline_oss20b/eval.json                deliverables/eval_baseline_oss20b.json 2>/dev/null || true
zip -r full_run_smoke_deliverables_oss20b.zip deliverables artifacts/figures artifacts/rq2_lora_only_smoke_oss20b artifacts/rq2_replay_only_smoke_oss20b artifacts/rq2_hybrid_smoke_oss20b logs/smoke_baseline_oss20b
```

## 2. Expected Outputs

• artifacts/rq2_*_smoke_oss20b/summary.csv → FFI, Legacy per day, train seconds, buffer size \

• artifacts/figures/{ffi_over_days.png, legacy_over_days.png, final_day_summary.csv} \

• logs/smoke_baseline_oss20b/eval.json \

• full_run_smoke_deliverables_oss20b.zip (ready to share)

## 3. What This Repo Contains
```.
├── scripts/
│   ├── prepare_days.py         # Build 30 “days” + legacy QA from Wikitext
│   ├── eval_qa.py              # Baseline exact-match eval (no adapters)
│   ├── train_lora.py           # LoRA adapter training for a given day
│   ├── eval_with_adapter.py    # Eval base + adapter on FFI / Legacy sets
│   ├── day_loop.py             # 30-day driver: LoRA / Replay / Hybrid
│   └── analyze_rq2.py          # Plots + CSV summary
├── data/                       # (ignored) generated “days” + legacy
├── models/                     # (ignored) saved LoRA adapters
├── artifacts/                  # (ignored) CSVs + figures
├── logs/                       # (ignored) eval logs + debug samples
├── requirements.txt
└── README.md
```

## 4. Interpreting Results
• FFI (Freshness): should trend upward for LoRA and Hybrid \

• Legacy Accuracy: best preserved by Replay and Hybrid \

• final_day_summary.csv → captures end-state FFI/Legacy for each regimen \

• The three summary.csv files (one per regimen) are the raw, per-day evidence
