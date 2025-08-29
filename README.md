# RQ2 — 30-Day Freshness: Continual Learning Regimens (LoRA / Replay / Hybrid)

This repo reproduces the **RQ2** experiment from the thesis:
> _Designing Real-Time, Continual-Learning Data-Pipeline Architectures for Enterprise LLMs._

We simulate **30 “days”** of new facts (from Wikitext), and compare three update regimens:
- **LoRA-only** (cheap daily adapter)
- **Replay-only** (memory buffer)
- **Hybrid** (LoRA + Replay 1:1)

Each day we measure:
- **FFI (Freshness):** exact-match on day-new questions
- **Legacy Accuracy:** exact-match on a fixed, pre-update question set

Runs fine on a **Mac M1/M2** (MPS). Later we’ll containerize for AKS.

---

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
