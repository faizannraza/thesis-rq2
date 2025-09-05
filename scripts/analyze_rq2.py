# scripts/analyze_rq2.py
# Robust analyzer for RQ2 runs: tolerates header variants and produces
# ffi_over_days.png, legacy_over_days.png, final_day_summary.csv, all_runs_concat.csv

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import pandas as pd


def pretty_label(dir_name: str) -> str:
    name = dir_name.lower()
    if "lora_only" in name:
        base = "LoRA-only"
    elif "replay_only" in name:
        base = "Replay-only"
    elif "hybrid" in name:
        base = "Hybrid"
    else:
        base = dir_name
    if "smoke" in name:
        base += " (smoke)"
    return base


def discover_runs(artifacts_root: Path) -> list[tuple[str, Path]]:
    runs = []
    # Look for artifacts/rq2_*/summary.csv
    for p in artifacts_root.glob("rq2_*/summary.csv"):
        label = pretty_label(p.parent.name)
        runs.append((label, p))
    # Also pick up deeper layouts just in case (optional, non-breaking)
    for p in artifacts_root.rglob("summary.csv"):
        if p.parent.name.startswith("rq2_") and (p.parent, p) not in runs:
            runs.append((pretty_label(p.parent.name), p))
    # De-duplicate by full path
    seen = set()
    uniq = []
    for label, p in runs:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            uniq.append((label, p))
    return uniq


# ----- Normalization helpers -----
_VARIANTS = {
    "day": ["day", "Day"],
    "FFI": ["FFI", "ffi", "freshness", "Freshness"],
    "LegacyAcc": ["LegacyAcc", "Legacy", "legacy", "legacy_acc", "legacyacc"],
    # Below are optional (not required for plotting, but we keep if present)
    "train_seconds": ["train_seconds", "train_s", "train_secs", "train_time"],
    "n_train": ["n_train", "n_today", "ntoday", "n_examples"],
    "buffer_size": ["buffer_size", "buf_size", "buffer", "buf"],
    "regimen": ["regimen"],
}

def _find(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    return None

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Rename any variant columns to canonical names
    rename = {}
    for canonical, variants in _VARIANTS.items():
        found = _find(df, variants)
        if found and found != canonical:
            rename[found] = canonical
    if rename:
        df = df.rename(columns=rename)

    # Ensure required columns exist
    if "day" not in df.columns:
        # Fallback: create a 1..N index if day missing (shouldn't happen in our runs)
        df["day"] = range(1, len(df) + 1)
    if "FFI" not in df.columns:
        df["FFI"] = 0.0
    if "LegacyAcc" not in df.columns:
        df["LegacyAcc"] = 0.0

    # Type coercion
    df["day"] = pd.to_numeric(df["day"], errors="coerce").fillna(0).astype(int)
    for c in ["FFI", "LegacyAcc"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)

    return df


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = normalize_columns(df)
    # Keep any extra known columns if present; plotting only needs day/FFI/LegacyAcc
    return df


def main(args):
    artifacts_root = Path(args.artifacts_root)
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Build run map: either explicit mapping or auto-discovery
    if args.paths:
        run_map = []
        # format: "Label1=path/to/summary.csv,Label2=path2,..."
        for item in args.paths.split(","):
            if not item.strip():
                continue
            label, path = item.split("=", 1)
            run_map.append((label.strip(), Path(path.strip())))
    else:
        run_map = discover_runs(artifacts_root)

    if not run_map:
        print(
            "No runs found.\n"
            "Expected to find at least one summary.csv under artifacts/rq2_*/summary.csv.\n"
            "Example smoke paths:\n"
            "  artifacts/rq2_lora_only_smoke/summary.csv\n"
            "  artifacts/rq2_replay_only_smoke/summary.csv\n"
            "  artifacts/rq2_hybrid_smoke/summary.csv\n",
            file=sys.stderr,
        )
        sys.exit(1)

    frames = []
    for label, path in run_map:
        if not path.exists():
            print(f"[warn] missing: {path}", file=sys.stderr)
            continue
        try:
            df = load_summary(path)
        except Exception as e:
            print(f"[warn] failed to read {path}: {e}", file=sys.stderr)
            continue
        df["RegimenLabel"] = label
        frames.append(df)

    if not frames:
        raise SystemExit("Found run entries, but none of the summary.csv files could be loaded.")

    df = pd.concat(frames, ignore_index=True)

    # ---- Plots ----
    # FFI over days
    plt.figure()
    for k, g in df.groupby("RegimenLabel"):
        g = g.sort_values("day")
        plt.plot(g["day"], g["FFI"], label=k, marker="o")
    plt.xlabel("Day")
    plt.ylabel("FFI (Exact Match)")
    plt.title("Freshness Over Days")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(outdir / "ffi_over_days.png", dpi=180, bbox_inches="tight")

    # Legacy over days
    plt.figure()
    for k, g in df.groupby("RegimenLabel"):
        g = g.sort_values("day")
        plt.plot(g["day"], g["LegacyAcc"], label=k, marker="o")
    plt.xlabel("Day")
    plt.ylabel("Legacy Accuracy (Exact Match)")
    plt.title("Legacy Retention Over Days")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(outdir / "legacy_over_days.png", dpi=180, bbox_inches="tight")

    # Final-day comparison table
    last = (
        df.sort_values("day")
          .groupby("RegimenLabel")
          .tail(1)[["RegimenLabel", "FFI", "LegacyAcc"]]
          .reset_index(drop=True)
    )
    last.to_csv(outdir / "final_day_summary.csv", index=False)

    # Save a combined CSV for convenience
    df.to_csv(outdir / "all_runs_concat.csv", index=False)

    print("Analyzed runs:")
    for label, path in run_map:
        print(f" - {label}: {path}")
    print("Wrote figures to", outdir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts_root", default="artifacts", help="Root folder to search")
    ap.add_argument(
        "--paths",
        default="",
        help='Optional explicit mapping: "LoRA-only=artifacts/rq2_lora_only_smoke/summary.csv,Replay-only=...".',
    )
    ap.add_argument("--out_dir", default="artifacts/figures")
    args = ap.parse_args()
    main(args)
