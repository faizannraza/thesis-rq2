# scripts/analyze_rq2.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def load_summary(path):
    df = pd.read_csv(path)
    return df

def main(args):
    runs = {
        "LoRA-only": Path("artifacts/rq2_lora_only/summary.csv"),
        "Replay-only": Path("artifacts/rq2_replay_only/summary.csv"),
        "Hybrid": Path("artifacts/rq2_hybrid/summary.csv"),
    }
    frames = []
    for name, path in runs.items():
        if path.exists():
            df = load_summary(path)
            df["RegimenLabel"] = name
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    outdir = Path("artifacts/figures")
    outdir.mkdir(parents=True, exist_ok=True)

    # FFI over days
    plt.figure()
    for k, g in df.groupby("RegimenLabel"):
        g = g.sort_values("day")
        plt.plot(g["day"], g["FFI"], label=k, marker="o")
    plt.xlabel("Day"); plt.ylabel("FFI (Exact Match)")
    plt.title("Freshness Over 30 Days")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(outdir / "ffi_over_days.png", dpi=180, bbox_inches="tight")

    # Legacy over days
    plt.figure()
    for k, g in df.groupby("RegimenLabel"):
        g = g.sort_values("day")
        plt.plot(g["day"], g["LegacyAcc"], label=k, marker="o")
    plt.xlabel("Day"); plt.ylabel("Legacy Accuracy (Exact Match)")
    plt.title("Legacy Retention Over 30 Days")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(outdir / "legacy_over_days.png", dpi=180, bbox_inches="tight")

    # Final day comparison
    last = df.groupby(["RegimenLabel"])\
        .apply(lambda g: g.loc[g["day"].idxmax()][["FFI","LegacyAcc"]])\
        .reset_index()
    last.to_csv(outdir / "final_day_summary.csv", index=False)
    print("Wrote figures to", outdir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    main(args)
