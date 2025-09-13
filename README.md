# RQ2 — STAR vs Baselines on CPU-Only Laptops

This repo reproduces the **RQ2** experiment from the thesis:
> _Designing Real-Time, Continual-Learning Data-Pipeline Architectures for Enterprise LLMs._

This README is a ready-to-run guide for Faizan, Emma, and Mark to reproduce the full RQ2 experiment matrix (LoRA-only, Replay-only, Hybrid, and STAR) across three datasets and three model sizes on CPU-only laptops (macOS or Windows). No placeholders: all commands build real streams from public corpora, train/eval adapters, and produce figures + roll-up CSVs everyone pushes back to this branch.

---

## Contents
- [What we run & why](#what-we-run--why)
- [Hardware/OS assumptions](#hardwareos-assumptions)
- [Environment setup (everyone)](#environment-setup-everyone)
- [Model suite (run all 3)](#model-suite-run-all-3)
- [Dataset matrix & randomness](#dataset-matrix--randomness)
- [Faizan's role: D1 (Wikipedia-style, 10 days)](#faizans-role-d1-wikipediastyle-10-days)
- [Emma’s role: D2 (News-style, 10 days)](#emmas-role-d2-newsstyle-10-days)
- [Mark’s role: D3 (StackExchange-style, 7 days)](#marks-role-d3-stackexchange-style-7-days)
- [Aggregate plots & statistics](#aggregate-plots--statistics)
- [What to push back](#what-to-push-back)
- [Interpreting the results](#interpreting-the-results)
- [Quickstart (macOS Apple Silicon)](#quickstart-macos-apple-silicon)
---

## What we run & why

# Hardware/OS assumptions

# Environment setup (everyone)

# Model suite (run all 3)

# Dataset matrix & randomness

# Faizan's role: D1 (Wikipedia-style, 10 days)

# Emma’s role: D2 (News-style, 10 days)

# Mark’s role: D3 (StackExchange-style, 7 days)

# Aggregate plots & statistics

# What to push back

# Interpreting the results

## Quickstart (macOS Apple Silicon)

1. Create env
   ```bash
   conda create -y -n rq2 python=3.10
   conda activate rq2
   pip install --upgrade pip
Install Torch (macOS arm64 wheels include MPS)


pip install torch torchvision torchaudio
Install the rest


pip install -r requirements.txt
Project layout (already in repo)


.
├── scripts/
│   ├── prepare_days.py
│   ├── eval_qa.py
│   ├── train_lora.py
│   ├── eval_with_adapter.py
│   └── day_loop.py
├── data/         # (ignored) generated
├── models/       # (ignored) adapters saved here
├── artifacts/    # (ignored) CSV + figs
├── logs/         # (ignored)
├── requirements.txt
├── environment.yml
└── README.md
Generate the 30 “days”


python scripts/prepare_days.py --days_dir data/days --legacy_dir data/legacy --days 30
Baseline smoke (no updates)


export MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
python scripts/eval_qa.py \
  --model_name "$MODEL" \
  --qa_file data/days/qa_eval_day_01.jsonl \
  --out_dir logs/smoke_baseline
One-day LoRA sanity


python scripts/train_lora.py \
  --base_model "$MODEL" \
  --train_jsonl data/days/qa_train_day_01.jsonl \
  --out_dir models/adapters/lora_only/day_01 \
  --epochs 1 --bsz 1 --grad_accum 8 --lr 2e-4

# Evaluate FFI + Legacy
python scripts/eval_with_adapter.py \
  --base_model "$MODEL" \
  --adapter_dir models/adapters/lora_only/day_01 \
  --qa_file data/days/qa_eval_day_01.jsonl \
  --out_dir logs/ffi_day01

python scripts/eval_with_adapter.py \
  --base_model "$MODEL" \
  --adapter_dir models/adapters/lora_only/day_01 \
  --qa_file data/legacy/qa_legacy_holdout.jsonl \
  --out_dir logs/legacy_day01
3-day smoke for each regimen


# LoRA-only
python scripts/day_loop.py --base_model "$MODEL" \
  --regimen lora_only --max_days 3 \
  --out_root artifacts/rq2_lora_only_smoke \
  --adapters_root models/adapters \
  --days_dir data/days --legacy_eval data/legacy/qa_legacy_holdout.jsonl \
  --epochs_per_day 1 --bsz 1 --grad_accum 8 --max_eval 200

# Replay-only
python scripts/day_loop.py --base_model "$MODEL" \
  --regimen replay_only --max_days 3 \
  --out_root artifacts/rq2_replay_only_smoke \
  --adapters_root models/adapters \
  --days_dir data/days --legacy_eval data/legacy/qa_legacy_holdout.jsonl \
  --epochs_per_day 1 --bsz 1 --grad_accum 8 --max_eval 200

# Hybrid
python scripts/day_loop.py --base_model "$MODEL" \
  --regimen hybrid --max_days 3 \
  --out_root artifacts/rq2_hybrid_smoke \
  --adapters_root models/adapters \
  --days_dir data/days --legacy_eval data/legacy/qa_legacy_holdout.jsonl \
  --epochs_per_day 1 --bsz 1 --grad_accum 8 --max_eval 200
Full 30-day runs (overnight batches)


# LoRA-only
python scripts/day_loop.py --base_model "$MODEL" \
  --regimen lora_only --max_days 30 \
  --out_root artifacts/rq2_lora_only \
  --adapters_root models/adapters \
  --epochs_per_day 3 --bsz 1 --grad_accum 8 --max_eval 500

# Replay-only
python scripts/day_loop.py --base_model "$MODEL" \
  --regimen replay_only --max_days 30 \
  --out_root artifacts/rq2_replay_only \
  --adapters_root models/adapters \
  --epochs_per_day 3 --bsz 1 --grad_accum 8 --max_eval 500

# Hybrid
python scripts/day_loop.py --base_model "$MODEL" \
  --regimen hybrid --max_days 30 \
  --out_root artifacts/rq2_hybrid \
  --adapters_root models/adapters \
  --epochs_per_day 3 --bsz 1 --grad_accum 8 --max_eval 500
Plot + summarize

python scripts/analyze_rq2.py
# Figures land in artifacts/figures/
Notes
Keep small batch sizes and fp32 on MPS.

If you hit OOM: reduce --max_len 384, --epochs_per_day 2, or try Qwen/Qwen2.5-0.5B-Instruct.

We intentionally do not commit data, adapters, or artifacts.

Repro on Linux/Windows
Create conda env: conda env create -f environment.yml && conda activate rq2

Install torch/torchvision/torchaudio for your platform (see PyTorch site), then:

pip install -r requirements.txt
Follow the same steps (4–10) above.

Contact
Muhammad Faizan Raza (mfr5933@psu.edu)
