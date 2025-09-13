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
- [Quick sanity checks](#quick-sanity-checks)
- [Quickstart (macOS Apple Silicon)](#quickstart-macos-apple-silicon)
- [Re-running with another model](#re-running-with-another-model)
---

## What we run & why
We evaluate continual-learning regimens under a day-by-day stream:

1. **LoRA-only** — adapts on the current day only.

2. **Replay-only** — trains only on a (reservoir) replay buffer.

3. **Hybrid** — 1:1 mix of the day’s data and replay buffer.

4. **STAR** — Selective Temporal Adapter Routing: learns multiple day adapters, trains a small router to pick top-k adapters per query, and uses FAR (freshness-aware replay) to stabilize memory.

We run **3 shuffled streams × 2 seeds** per dataset to get variance; then we aggregate, compute CIs/paired tests, and visualize Pareto fronts (Freshness vs Legacy retention).

---

## Hardware/OS assumptions
• CPU-only laptops (macOS or Windows).

• Optional on Apple Silicon: MPS acceleration (speeds up training; still CPU-safe).

• No CUDA required.

Expect runs to be slow but feasible; keep Terminal open.

---

## Environment setup (everyone)
**macOS (bash/zsh)**
   ```bash
   # 0) clone the branch (if you haven’t)
git clone https://github.com/muhammadfaizanraza/continual-learning-regimen-test-exp-rq2.git
cd continual-learning-regimen-test-exp-rq2
git checkout rq2-CPU-STAR

# 1) venv
python3 -m venv venv
source venv/bin/activate

# 2) deps (safe to re-run)
pip install --upgrade pip
pip install -r requirements.txt
# (if requirements.txt is minimal):
pip install scikit-learn==1.4.2 sentence-transformers==2.5.1

# 3) caches
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"
export HF_HUB_ENABLE_HF_TRANSFER=1

# 4) runtime knobs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0   # if you enable MPS

# 5) device flags
# CPU default:
export DEVICE_FLAG=""
export FP16_FLAG=""
# If on Apple Silicon and you want acceleration:
# export DEVICE_FLAG="--device mps"
```
**Windows (PowerShell)**
```bash
# 0) clone the branch (if you haven’t)
git clone https://github.com/muhammadfaizanraza/continual-learning-regimen-test-exp-rq2.git
cd continual-learning-regimen-test-exp-rq2
git checkout rq2-CPU-STAR

# 1) venv
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2) deps
pip install --upgrade pip
pip install -r requirements.txt
pip install scikit-learn==1.4.2 sentence-transformers==2.5.1

# 3) caches
$env:HF_HOME="$HOME\.cache\huggingface"
$env:TRANSFORMERS_CACHE=$env:HF_HOME
$env:HF_DATASETS_CACHE=$env:HF_HOME
$env:HF_HUB_ENABLE_HF_TRANSFER="1"

# 4) device flags (CPU only on Windows)
$env:DEVICE_FLAG=""
$env:FP16_FLAG=""
```
---
## Model suite (run all 3)
We compare three CPU-friendly instruction-tuned models:

1. **TinyLlama/TinyLlama-1.1B-Chat-v1.0** (~1.1B, small-capacity baseline)

2. **Qwen/Qwen2.5-1.5B-Instruct** (~1.5B, strong small model; our default)

3. **Qwen/Qwen2.5-3B-Instruct** (~3B, larger; slower on CPU/MPS but feasible for our matrix)

> _We do not run 7B on CPU in this branch. It’s impractical for the full matrix.

For each block below, set `MODEL` and `MODEL_TAG` before running the dataset loops:

**macOS/Linux (bash/zsh)**
```bash
# Choose one model at a time, then repeat the dataset loops for all three.

# (A) 1.1B
export MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"; export MODEL_TAG="M1p1B_TinyLlama"

# (B) 1.5B (recommended default)
# export MODEL="Qwen/Qwen2.5-1.5B-Instruct"; export MODEL_TAG="M1p5B_Qwen25"

# (C) 3B
# export MODEL="Qwen/Qwen2.5-3B-Instruct"; export MODEL_TAG="M3B_Qwen25"
```
**Windows (PowerShell)**
```bash
# (A) 1.1B
$env:MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"; $env:MODEL_TAG="M1p1B_TinyLlama"

# (B) 1.5B (recommended default)
# $env:MODEL="Qwen/Qwen2.5-1.5B-Instruct"; $env:MODEL_TAG="M1p5B_Qwen25"

# (C) 3B
# $env:MODEL="Qwen/Qwen2.5-3B-Instruct"; $env:MODEL_TAG="M3B_Qwen25"
```
All **outputs** will be prefixed with the chosen `$MODEL_TAG`, e.g. `artifacts/D1_M1p5B_Qwen25_s11_seed1_lora/...`

---
## Dataset matrix & randomness
We run the same regimen matrix across **three datasets**:

• **D1 (Faizan)**: Wikipedia-style open-domain QA, **10 days**

• **D2 (Emma)**: News-style temporal QA, **10 days**

• **D3 (Mark)**: StackExchange-style QA, **7 days**

**Randomness within datasets.**
We create **3 shuffled streams** per dataset by changing `--seed`. Each call to `prepare_days.py` downloads/builds from the real public corpus for that dataset and shuffles into day bins. You’ll see outputs like `data/days_s11`, `data/days_s12`, `data/days_s13`, each with `qa_train_day_XX.jsonl` and `qa_eval_day_XX.jsonl`. The **legacy holdout** (`qa_legacy_holdout.jsonl`) is built from a fixed corpus slice.

> _Tip: verify --seed support with python scripts/prepare_days.py -h. This branch’s script takes --days, --seed, and --source.

**Replay buffer sampling.**
During training, the buffer is updated daily and sub-sampled with **reservoir sampling** (see `scripts/day_loop.py`), governed by `--replay_rate` and `--min_replay_cap`.

---

# Faizan's role: D1 (Wikipedia-style, 10 days)
< -Run these for each model (set `MODEL`/`MODEL_TAG`, then run steps 1–3).
macOS shows bash loops; Windows shows PowerShell after each block.

**1) Build 3 shuffled 10-day streams**

**macOS / Linux**
```bash
python scripts/prepare_days.py --source wiki_qa --days 10 --seed 101 --days_dir data/days_s11 --legacy_dir data/legacy_s11
python scripts/prepare_days.py --source wiki_qa --days 10 --seed 202 --days_dir data/days_s12 --legacy_dir data/legacy_s12
python scripts/prepare_days.py --source wiki_qa --days 10 --seed 303 --days_dir data/days_s13 --legacy_dir data/legacy_s13
```
**Windows (PowerShell)**
```bash
python scripts/prepare_days.py --source wiki_qa --days 10 --seed 101 --days_dir data/days_s11 --legacy_dir data/legacy_s11
python scripts/prepare_days.py --source wiki_qa --days 10 --seed 202 --days_dir data/days_s12 --legacy_dir data/legacy_s12
python scripts/prepare_days.py --source wiki_qa --days 10 --seed 303 --days_dir data/days_s13 --legacy_dir data/legacy_s13
```
**2) Baselines: LoRA / Replay / Hybrid (2 seeds each)**

**macOS / Linux**
```bash
for S in 11 12 13; do
  for SEED in 1 2; do
    # LoRA-only
    python scripts/day_loop.py --base_model "$MODEL" --regimen lora_only --max_days 10 \
      --days_dir data/days_s$S --legacy_eval data/legacy_s$S/qa_legacy_holdout.jsonl \
      --out_root artifacts/D1_${MODEL_TAG}_s$S_seed${SEED}_lora --adapters_root models/adapters \
      --epochs_per_day 2 --bsz 1 --grad_accum 8 --max_eval 500 $DEVICE_FLAG $FP16_FLAG

    # Replay-only
    python scripts/day_loop.py --base_model "$MODEL" --regimen replay_only --max_days 10 \
      --days_dir data/days_s$S --legacy_eval data/legacy_s$S/qa_legacy_holdout.jsonl \
      --out_root artifacts/D1_${MODEL_TAG}_s$S_seed${SEED}_replay --adapters_root models/adapters \
      --epochs_per_day 2 --bsz 1 --grad_accum 8 --max_eval 500 $DEVICE_FLAG $FP16_FLAG

    # Hybrid
    python scripts/day_loop.py --base_model "$MODEL" --regimen hybrid --max_days 10 \
      --days_dir data/days_s$S --legacy_eval data/legacy_s$S/qa_legacy_holdout.jsonl \
      --out_root artifacts/D1_${MODEL_TAG}_s$S_seed${SEED}_hybrid --adapters_root models/adapters \
      --epochs_per_day 2 --bsz 1 --grad_accum 8 --max_eval 500 $DEVICE_FLAG $FP16_FLAG
  done
done
```
**Windows (PowerShell)**
```bash
foreach ($S in 11,12,13) {
  foreach ($SEED in 1,2) {
    python scripts/day_loop.py --base_model $env:MODEL --regimen lora_only --max_days 10 `
      --days_dir "data/days_s$S" --legacy_eval "data/legacy_s$S/qa_legacy_holdout.jsonl" `
      --out_root "artifacts/D1_${env:MODEL_TAG}_s$S_seed${SEED}_lora" --adapters_root models/adapters `
      --epochs_per_day 2 --bsz 1 --grad_accum 8 --max_eval 500 $env:DEVICE_FLAG $env:FP16_FLAG

    python scripts/day_loop.py --base_model $env:MODEL --regimen replay_only --max_days 10 `
      --days_dir "data/days_s$S" --legacy_eval "data/legacy_s$S/qa_legacy_holdout.jsonl" `
      --out_root "artifacts/D1_${env:MODEL_TAG}_s$S_seed${SEED}_replay" --adapters_root models/adapters `
      --epochs_per_day 2 --bsz 1 --grad_accum 8 --max_eval 500 $env:DEVICE_FLAG $env:FP16_FLAG

    python scripts/day_loop.py --base_model $env:MODEL --regimen hybrid --max_days 10 `
      --days_dir "data/days_s$S" --legacy_eval "data/legacy_s$S/qa_legacy_holdout.jsonl" `
      --out_root "artifacts/D1_${env:MODEL_TAG}_s$S_seed${SEED}_hybrid" --adapters_root models/adapters `
      --epochs_per_day 2 --bsz 1 --grad_accum 8 --max_eval 500 $env:DEVICE_FLAG $env:FP16_FLAG
  }
}
```
**3) STAR (2 seeds each)**

**macOS / Linux**
```bash
for S in 11 12 13; do
  for SEED in 1 2; do
    python scripts/day_loop_star.py --base_model "$MODEL" --max_days 10 \
      --days_dir data/days_s$S --legacy_eval data/legacy_s$S/qa_legacy_holdout.jsonl \
      --out_root artifacts/D1_${MODEL_TAG}_s$S_seed${SEED}_star --adapters_root models/adapters \
      --epochs_per_day 2 --bsz 1 --grad_accum 8 --max_eval 500 \
      --pool_size 8 --top_k 2 --router_epochs 8 --router_lr 1e-3 \
      --replay_rate 0.03 --min_replay_cap 128 --lam_h 1.0 --lam_n 0.5 --lam_t 0.2 --far_gamma 0.1 \
      $DEVICE_FLAG $FP16_FLAG
  done
done
```
**Windows (PowerShell)**
```bash
foreach ($S in 11,12,13) {
  foreach ($SEED in 1,2) {
    python scripts/day_loop_star.py --base_model $env:MODEL --max_days 10 `
      --days_dir "data/days_s$S" --legacy_eval "data/legacy_s$S/qa_legacy_holdout.jsonl" `
      --out_root "artifacts/D1_${env:MODEL_TAG}_s$S_seed${SEED}_star" --adapters_root models/adapters `
      --epochs_per_day 2 --bsz 1 --grad_accum 8 --max_eval 500 `
      --pool_size 8 --top_k 2 --router_epochs 8 --router_lr 1e-3 `
      --replay_rate 0.03 --min_replay_cap 128 --lam_h 1.0 --lam_n 0.5 --lam_t 0.2 --far_gamma 0.1 `
      $env:DEVICE_FLAG $env:FP16_FLAG
  }
}
```

## Emma’s role: D2 (News-style, 10 days)
< _Run per model (set MODEL/MODEL_TAG first), then:

**1) Build 3 shuffled 10-day streams**
**macOS / Linux**
```bash
python scripts/prepare_days.py --source news_qa --days 10 --seed 1111 --days_dir data_d2/days_s21 --legacy_dir data_d2/legacy_s21
python scripts/prepare_days.py --source news_qa --days 10 --seed 2222 --days_dir data_d2/days_s22 --legacy_dir data_d2/legacy_s22
python scripts/prepare_days.py --source news_qa --days 10 --seed 3333 --days_dir data_d2/days_s23 --legacy_dir data_d2/legacy_s23
```
**Windows (PowerShell)** – same three lines with backslashes replaced by `\`.

**2) Baselines + STAR (2 seeds each)**
**macOS / Linux**
```bash
for S in 21 22 23; do
  for SEED in 1 2; do
    python scripts/day_loop.py --base_model "$MODEL" --regimen lora_only --max_days 10 \
      --days_dir data_d2/days_s$S --legacy_eval data_d2/legacy_s$S/qa_legacy_holdout.jsonl \
      --out_root artifacts/D2_${MODEL_TAG}_s$S_seed${SEED}_lora --adapters_root models/adapters \
      --epochs_per_day 2 --bsz 1 --grad_accum 8 --max_eval 500 $DEVICE_FLAG $FP16_FLAG

    python scripts/day_loop.py --base_model "$MODEL" --regimen replay_only --max_days 10 \
      --days_dir data_d2/days_s$S --legacy_eval data_d2/legacy_s$S/qa_legacy_holdout.jsonl \
      --out_root artifacts/D2_${MODEL_TAG}_s$S_seed${SEED}_replay --adapters_root models/adapters \
      --epochs_per_day 2 --bsz 1 --grad_accum 8 --max_eval 500 $DEVICE_FLAG $FP16_FLAG

    python scripts/day_loop.py --base_model "$MODEL" --regimen hybrid --max_days 10 \
      --days_dir data_d2/days_s$S --legacy_eval data_d2/legacy_s$S/qa_legacy_holdout.jsonl \
      --out_root artifacts/D2_${MODEL_TAG}_s$S_seed${SEED}_hybrid --adapters_root models/adapters \
      --epochs_per_day 2 --bsz 1 --grad_accum 8 --max_eval 500 $DEVICE_FLAG $FP16_FLAG
  done
  for SEED in 1 2; do
    python scripts/day_loop_star.py --base_model "$MODEL" --max_days 10 \
      --days_dir data_d2/days_s$S --legacy_eval data_d2/legacy_s$S/qa_legacy_holdout.jsonl \
      --out_root artifacts/D2_${MODEL_TAG}_s$S_seed${SEED}_star --adapters_root models/adapters \
      --epochs_per_day 2 --bsz 1 --grad_accum 8 --max_eval 500 \
      --pool_size 8 --top_k 2 --router_epochs 8 --router_lr 1e-3 \
      $DEVICE_FLAG $FP16_FLAG
  done
done
```
**Windows (PowerShell)** – translate to `foreach` as in D1.

## Mark’s role: D3 (StackExchange-style, 7 days)
< _Run per model (set `MODEL`/`MODEL_TAG` first), then:

**1) Build 3 shuffled 7-day streams**
```bash
python scripts/prepare_days.py --source stack_qa --days 7 --seed 5151 --days_dir data_d3/days_s31 --legacy_dir data_d3/legacy_s31
python scripts/prepare_days.py --source stack_qa --days 7 --seed 6161 --days_dir data_d3/days_s32 --legacy_dir data_d3/legacy_s32
python scripts/prepare_days.py --source stack_qa --days 7 --seed 7171 --days_dir data_d3/days_s33 --legacy_dir data_d3/legacy_s33
```
**2) Baselines + STAR (2 seeds each)**
**macOS / Linux**
```bash
for S in 31 32 33; do
  for SEED in 1 2; do
    python scripts/day_loop.py --base_model "$MODEL" --regimen lora_only --max_days 7 \
      --days_dir data_d3/days_s$S --legacy_eval data_d3/legacy_s$S/qa_legacy_holdout.jsonl \
      --out_root artifacts/D3_${MODEL_TAG}_s$S_seed${SEED}_lora --adapters_root models/adapters \
      --epochs_per_day 2 --bsz 1 --grad_accum 8 --max_eval 500 $DEVICE_FLAG $FP16_FLAG

    python scripts/day_loop.py --base_model "$MODEL" --regimen replay_only --max_days 7 \
      --days_dir data_d3/days_s$S --legacy_eval data_d3/legacy_s$S/qa_legacy_holdout.jsonl \
      --out_root artifacts/D3_${MODEL_TAG}_s$S_seed${SEED}_replay --adapters_root models/adapters \
      --epochs_per_day 2 --bsz 1 --grad_accum 8 --max_eval 500 $DEVICE_FLAG $FP16_FLAG

    python scripts/day_loop.py --base_model "$MODEL" --regimen hybrid --max_days 7 \
      --days_dir data_d3/days_s$S --legacy_eval data_d3/legacy_s$S/qa_legacy_holdout.jsonl \
      --out_root artifacts/D3_${MODEL_TAG}_s$S_seed${SEED}_hybrid --adapters_root models/adapters \
      --epochs_per_day 2 --bsz 1 --grad_accum 8 --max_eval 500 $DEVICE_FLAG $FP16_FLAG

    python scripts/day_loop_star.py --base_model "$MODEL" --max_days 7 \
      --days_dir data_d3/days_s$S --legacy_eval data_d3/legacy_s$S/qa_legacy_holdout.jsonl \
      --out_root artifacts/D3_${MODEL_TAG}_s$S_seed${SEED}_star --adapters_root models/adapters \
      --epochs_per_day 2 --bsz 1 --grad_accum 8 --max_eval 500 \
      --pool_size 8 --top_k 2 --router_epochs 8 --router_lr 1e-3 \
      $DEVICE_FLAG $FP16_FLAG
  done
done
```
**Windows (PowerShell)** – translate to `foreach` as in D1.

## Aggregate plots & statistics
Once all three of us have run all models, pull everyone’s `artifacts/` into this branch, then
```bash
# 1) Aggregate and plot
python scripts/analyze_star_and_baselines.py --glob "artifacts/D*/**" --out artifacts/figures_star

# 2) Compute confidence intervals, paired tests, Pareto (FFI vs Legacy)
python scripts/stats_star.py --in_csv artifacts/figures_star/all_runs_concat.csv \
  --out_dir artifacts/figures_star --alpha 0.05
```
**Outputs:**

• `artifacts/figures_star/ffi_em_all_runs.png` — FFI (EM) curves for every run.

• `artifacts/figures_star/all_runs_concat.csv` — concatenated summaries (everything you need for stats).

• `artifacts/figures_star/stats_summary.md` — CIs and paired tests (STAR vs baselines).

• `artifacts/figures_star/pareto_ffi_legacy.png` — Pareto scatter with iso-cost hints.

## What to push back
We use Git LFS for large files. From your repo root:

```bash
# If not already configured:
git lfs install
git lfs track "*.jsonl" "*.csv" "*.pt" "*.bin" "*.safetensors" "*.zip" "*.parquet"

git add artifacts data/*.md  # add your newly created artifacts and any notes
git add artifacts/**/summary.csv artifacts/**/eval.json artifacts/figures_star/* -f
git commit -m "Add RQ2 CPU STAR matrix results for ${MODEL_TAG} (D1/D2/D3)"
git push -u origin rq2-CPU-STAR
```
**Keep:**

• `artifacts/D*_M*/sXX_seedY_{lora|replay|hybrid|star}/summary.csv`

• Per-day eval JSONs under each run (`.../ffi/day_*/eval.json`, `.../legacy/day_*/eval.json`)

• `artifacts/figures_star/*` (plots + stats + combined CSV)

**Don’t push**: raw model weights not needed for reproduction (LFS rules already ignore most; if in doubt, keep).

## Interpreting the results
Each run produces a `summary.csv` with per-day metrics:

• **FFI_EM / FFI_F1 / FFI_nEM** — accuracy on **fresh** day-eval (adaptation).

• **Legacy_EM / Legacy_F1 / Legacy_nEM** — accuracy on the **unchanged** legacy holdout (retention).

• **train_seconds / n_train / buffer_size** — compute & replay footprint for cost analysis.


Look for:

• **Freshness (FFI):** STAR should match or exceed Hybrid/LoRA on most days.

• **Legacy:** STAR should avoid the sharper drops seen in LoRA-only when new info conflicts.

• **Stability:** lower variance day-to-day under STAR due to routing.

• **Cost:** STAR’s `train_seconds` near LoRA + small router; Replay-only may increase cost via big buffers without freshness gains.


**Model-size trends (typical):**

• **1.1B (TinyLlama)**: lower ceilings; replay stabilizes but can mute freshness; STAR often helps most here.

• **1.5B (Qwen2.5-1.5B)**: strongest overall trade-off; our default comparisons.

• **3B (Qwen2.5-3B)**: higher FFI and Legacy but slower; STAR typically remains Pareto-competitive.

## Quick sanity checks

• Help pages:
```bash
python scripts/prepare_days.py -h
python scripts/day_loop.py -h
python scripts/day_loop_star.py -h
```

• If Apple Silicon, confirm MPS:
```bash
python - <<'PY'
import torch; print("MPS available:", torch.backends.mps.is_available())
PY
```

## Re-running with another model
Just set a different `MODEL`/`MODEL_TAG` and re-run the same **D1/D2/D3** blocks. You’ll get separate run folders, e.g.:
```bash
artifacts/
  D1_M1p1B_TinyLlama_s11_seed1_lora/
  D1_M1p5B_Qwen25_s11_seed1_hybrid/
  D2_M3B_Qwen25_s23_seed2_star/
  ...
```
The analyzer/stats scripts consume **all of them at once** via the glob.

If anything is unclear or you see a CLI flag error, run the `-h` help for that script and adjust as indicated above. Otherwise, copy-paste these blocks verbatim and you’ll reproduce the full RQ2 matrix across **three datasets × three models × (3 shuffles × 2 seeds) × (4 regimens)** on CPU-only laptops.

If you need any help, contact Muhammad Faizan Raza (mfr5933@psu.edu)
