# scripts/analyze_star_and_baselines.py
import argparse, pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def collect(root_glob):
    rows = []
    for p in Path(".").glob(root_glob):
        csv = p/"summary.csv"
        if not csv.exists(): continue
        df = pd.read_csv(csv)
        label = p.name
        df["Run"] = label
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="artifacts/D*/**")  # pick up all
    ap.add_argument("--out", default="artifacts/figures_star")
    args = ap.parse_args()
    df = collect(args.glob)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    if df.empty: raise SystemExit("No summary.csv found")

    # harmonize: if FFI present, rename; else compute from EM columns
    if "FFI" in df.columns:
        df["FFI_EM"] = df["FFI"]; df["Legacy_EM"] = df["LegacyAcc"]

    # plot means across runs for day vs FFI_EM
    plt.figure()
    for run, g in df.groupby("Run"):
        plt.plot(g["day"], g["FFI_EM"], alpha=0.4, label=run)
    plt.xlabel("Day"); plt.ylabel("FFI (EM)"); plt.title("FFI across runs")
    plt.legend(fontsize=6, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.savefig(Path(args.out)/"ffi_em_all_runs.png", dpi=180, bbox_inches="tight")

    df.to_csv(Path(args.out)/"all_runs_concat.csv", index=False)
    print("Wrote plots and CSV to", args.out)