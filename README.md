RQ2 — 30-Day Freshness (GPU Edition)

Continual Learning Regimens: LoRA / Replay / Hybrid

This branch reproduces the RQ2 experiment from the thesis:

Designing Real-Time, Continual-Learning Data-Pipeline Architectures for Enterprise LLMs

We simulate 30 “days” of new facts (from Wikitext) and compare three update regimens:

LoRA-only – cheap daily adapters

Replay-only – memory buffer for retention

Hybrid – LoRA + small replay (≈1:1 mix)

Daily metrics:

FFI (Freshness): exact-match on day-new questions

Legacy Accuracy: exact-match on a fixed, pre-update question set

This branch targets NVIDIA GPUs (CUDA). It also runs on CPU/MPS if a GPU isn’t available.

0) Repo layout
.
├── scripts/
│   ├── prepare_days.py          # build 30 “days” of cloze QA
│   ├── eval_qa.py               # baseline eval (no adapters)
│   ├── train_lora.py            # train LoRA adapters (GPU/CPU/MPS)
│   ├── eval_with_adapter.py     # eval base + adapter (FFI/Legacy)
│   └── day_loop.py              # 30-day driver for each regimen
├── requirements.txt             # common deps
├── requirements-gpu.txt         # +optional GPU extras (bitsandbytes, etc.)
├── environment.yml              # optional conda env
├── data/                        # (gitignored) generated datasets
├── models/                      # (gitignored) adapters saved here
├── artifacts/                   # (gitignored) CSVs & plots
├── logs/                        # (gitignored) run logs & debug samples
└── README.md


We do not commit data/, models/, artifacts/, or logs/ — they can be large.

1) Environment (GPU first, CPU/MPS fallback)
1.1 Create & activate env
conda create -y -n rq2-gpu python=3.10
conda activate rq2-gpu
python -V

1.2 Install PyTorch for your platform

CUDA (Linux with NVIDIA GPU) – install from PyTorch:

# Example for CUDA 12.1 wheels; check pytorch.org for the latest line
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio


macOS (Apple Silicon, MPS) or CPU only:

pip install --upgrade pip
pip install torch torchvision torchaudio


Sanity check:

python - <<'PY'
import torch, platform
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("MPS available:", torch.backends.mps.is_available())
print("Device chosen:", "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(platform.platform())
PY

1.3 Install the rest
# Base requirements
pip install -r requirements.txt

# Optional GPU extras (Linux only; ignore on macOS):
# - bitsandbytes: 4/8-bit optimizers for QLoRA (optional; our default is standard LoRA)
# - xformers: memory efficient attention (optional)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
  pip install -r requirements-gpu.txt || true
fi


If bitsandbytes fails on your GPU, skip it — this pipeline doesn’t require it.

2) Pick a model

Default (safe on almost any GPU/CPU):

export MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"


Other options:

Qwen/Qwen2.5-0.5B-Instruct (even lighter)

Llama-3.x 8B (requires ~16–24GB GPU; adjust batch/seq len)

openai/gpt-oss-20b (H100 class; pass chat template flags and larger VRAM)

Chat-tuned models: when evaluating, use the tag prompt (default) or, if your script supports it, the --use_chat_template flag.

3) Generate the 30 “days” of data
python scripts/prepare_days.py --days_dir data/days --legacy_dir data/legacy --days 30


Outputs:

data/legacy/qa_legacy_holdout.jsonl

data/days/qa_train_day_01.jsonl … qa_train_day_30.jsonl

data/days/qa_eval_day_01.jsonl … qa_eval_day_30.jsonl

4) Baseline smoke (no adapters)
python scripts/eval_qa.py \
  --model_name "$MODEL" \
  --qa_file data/days/qa_eval_day_01.jsonl \
  --out_dir logs/smoke_baseline \
  --max_eval 300 \
  --max_new_tokens 8


If your model is chat-tuned and your script supports it:

# Add this flag if eval_qa.py supports it
# --use_chat_template


Expect a small but non-zero exact-match (it’s cloze-style).

5) One-day LoRA sanity (GPU)

Train a day-01 adapter (GPU if available; the scripts auto-choose cuda → mps → cpu):

python scripts/train_lora.py \
  --base_model "$MODEL" \
  --train_jsonl data/days/qa_train_day_01.jsonl \
  --out_dir models/adapters/lora_only/day_01 \
  --epochs 1 --bsz 1 --grad_accum 8 --lr 2e-4


Evaluate FFI on day-01:

python scripts/eval_with_adapter.py \
  --base_model "$MODEL" \
  --adapter_dir models/adapters/lora_only/day_01 \
  --qa_file data/days/qa_eval_day_01.jsonl \
  --out_dir logs/ffi_day01 \
  --max_eval 300 --max_new_tokens 8


Evaluate Legacy on the fixed set:

python scripts/eval_with_adapter.py \
  --base_model "$MODEL" \
  --adapter_dir models/adapters/lora_only/day_01 \
  --qa_file data/legacy/qa_legacy_holdout.jsonl \
  --out_dir logs/legacy_day01 \
  --max_eval 300 --max_new_tokens 8

6) Three 3-day smoke runs (GPU-accelerated)

These confirm the loop and produce summary.csv for each regimen.

6.1 LoRA-only (3 days)
python scripts/day_loop.py --base_model "$MODEL" \
  --regimen lora_only --max_days 3 \
  --out_root artifacts/rq2_lora_only_smoke \
  --adapters_root models/adapters \
  --days_dir data/days \
  --legacy_eval data/legacy/qa_legacy_holdout.jsonl \
  --epochs_per_day 1 --bsz 1 --grad_accum 8 --max_eval 200

6.2 Replay-only (3 days)
python scripts/day_loop.py --base_model "$MODEL" \
  --regimen replay_only --max_days 3 \
  --out_root artifacts/rq2_replay_only_smoke \
  --adapters_root models/adapters \
  --days_dir data/days \
  --legacy_eval data/legacy/qa_legacy_holdout.jsonl \
  --epochs_per_day 1 --bsz 1 --grad_accum 8 --max_eval 200

6.3 Hybrid (3 days)
python scripts/day_loop.py --base_model "$MODEL" \
  --regimen hybrid --max_days 3 \
  --out_root artifacts/rq2_hybrid_smoke \
  --adapters_root models/adapters \
  --days_dir data/days \
  --legacy_eval data/legacy/qa_legacy_holdout.jsonl \
  --epochs_per_day 1 --bsz 1 --grad_accum 8 --max_eval 200


Each run writes summary.csv to its artifacts/... directory.

7) Full 30-day runs (overnight batches)
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


Tip (GPU): on big models, try --max_len 384, --epochs_per_day 2, or a smaller model.

8) Plot & summarize
python scripts/analyze_rq2.py


Outputs go to:

artifacts/figures/
  ├── ffi_over_days.png
  ├── legacy_over_days.png
  └── final_day_summary.csv

9) Switching to bigger models (optional)

Example for Llama-3-8B (24GB GPU recommended) or gpt-oss-20b (H100 class):

Reduce sequence length: --max_len 384

Lower per-device batch: --bsz 1 and increase --grad_accum

Prefer bf16 (handled automatically if your GPU supports it)

For chat-tuned models, enable chat template if your eval script provides --use_chat_template

Example (LoRA day-01, cautious settings):

export MODEL="meta-llama/Llama-3.1-8B-Instruct"

python scripts/train_lora.py \
  --base_model "$MODEL" \
  --train_jsonl data/days/qa_train_day_01.jsonl \
  --out_dir models/adapters/lora_only/day_01 \
  --epochs 1 --bsz 1 --grad_accum 16 --lr 2e-4 --max_len 384

10) What “success” looks like

Each regimen folder (e.g., artifacts/rq2_hybrid/) contains:

summary.csv with rows for day, regimen, FFI, LegacyAcc, train_seconds, n_train, buffer_size

Subfolders ffi/day_XX/eval.json and legacy/day_XX/eval.json

artifacts/figures/ contains the plots

Your Hybrid curve typically climbs in FFI faster than Replay-only and retains Legacy better than LoRA-only.

11) Troubleshooting

A) “No objects to concatenate” (analyze script)
You didn’t run the regimen loops that create summary.csv. Make sure these files exist:

artifacts/rq2_lora_only/summary.csv
artifacts/rq2_replay_only/summary.csv
artifacts/rq2_hybrid/summary.csv


Then re-run:

python scripts/analyze_rq2.py


B) CUDA OOM
Use a smaller model, reduce --max_len to 384, drop --epochs_per_day, or increase --grad_accum with --bsz 1.

C) bitsandbytes install fails
Skip it (not required). Our default LoRA uses standard AdamW.

D) Chat models return zero exact-match
Use the plain tag prompt (default) or enable --use_chat_template if supported by your script, and reduce --max_new_tokens to 6–12 for cloze answers.

12) Reproduce in one go (GPU)
# 1) env
conda create -y -n rq2-gpu python=3.10 && conda activate rq2-gpu
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
pip install -r requirements.txt
if [[ "$OSTYPE" == "linux-gnu"* ]]; then pip install -r requirements-gpu.txt || true; fi

# 2) data
python scripts/prepare_days.py --days_dir data/days --legacy_dir data/legacy --days 30

# 3) model
export MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# 4) 3-day smokes (produce summary.csv’s)
python scripts/day_loop.py --base_model "$MODEL" --regimen lora_only   --max_days 3 --out_root artifacts/rq2_lora_only_smoke   --adapters_root models/adapters --days_dir data/days --legacy_eval data/legacy/qa_legacy_holdout.jsonl --epochs_per_day 1 --bsz 1 --grad_accum 8 --max_eval 200
python scripts/day_loop.py --base_model "$MODEL" --regimen replay_only --max_days 3 --out_root artifacts/rq2_replay_only_smoke --adapters_root models/adapters --days_dir data/days --legacy_eval data/legacy/qa_legacy_holdout.jsonl --epochs_per_day 1 --bsz 1 --grad_accum 8 --max_eval 200
python scripts/day_loop.py --base_model "$MODEL" --regimen hybrid      --max_days 3 --out_root artifacts/rq2_hybrid_smoke      --adapters_root models/adapters --days_dir data/days --legacy_eval data/legacy/qa_legacy_holdout.jsonl --epochs_per_day 1 --bsz 1 --grad_accum 8 --max_eval 200

# 5) 30-day full runs (overnight)
python scripts/day_loop.py --base_model "$MODEL" --regimen lora_only   --max_days 30 --out_root artifacts/rq2_lora_only  --adapters_root models/adapters --epochs_per_day 3 --bsz 1 --grad_accum 8 --max_eval 500
python scripts/day_loop.py --base_model "$MODEL" --regimen replay_only --max_days 30 --out_root artifacts/rq2_replay_only --adapters_root models/adapters --epochs_per_day 3 --bsz 1 --grad_accum 8 --max_eval 500
python scripts/day_loop.py --base_model "$MODEL" --regimen hybrid      --max_days 30 --out_root artifacts/rq2_hybrid      --adapters_root models/adapters --epochs_per_day 3 --bsz 1 --grad_accum 8 --max_eval 500

# 6) plots
python scripts/analyze_rq2.py

13) Contact

Muhammad Faizan Raza – mfr5933@psu.edu

PRs and issues welcome on this branch.

Notes:

This branch is GPU-first but will run on MPS/CPU automatically with smaller models.

When IT finalizes the cluster, we’ll containerize day_loop.py and run the exact same logic on AKS/Argo; no code changes to experiment logic.
