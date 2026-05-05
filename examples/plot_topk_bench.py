"""Plot top-k kernel benchmark results (32k inputs only).

Inputs : topk_bench_20260505_075252.csv
Outputs:
    topk_speedup.pdf    Speedup of v2 / approx (and remap variants) over the
                        two baselines, broken down by k, distribution, model.
    topk_correctness.pdf Recall comparison and per-config recall vs latency.

Baselines: topk_output (full-sort) and topk_output_sglang_ori (SGLang).
Best-of  : per-config minimum-latency variant inside each kernel family
           (best alpha for approx, best mapping for *_remap).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

CSV = Path("/data/datasets/xinrui/topk_bench_20260505_075252.csv")
OUT = Path("/data/datasets/xinrui/topk_kernels.pdf")

# ----- conference-quality matplotlib defaults --------------------------------
mpl.rcParams.update({
    "font.family"      : "serif",
    "font.serif"       : ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size"        : 9.5,
    "axes.titlesize"   : 10.5,
    "axes.labelsize"   : 10,
    "xtick.labelsize"  : 9,
    "ytick.labelsize"  : 9,
    "legend.fontsize"  : 8.5,
    "axes.linewidth"   : 0.8,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "pdf.fonttype"     : 42,
    "ps.fonttype"      : 42,
})

KEY = ["model", "batch_size", "distribution", "input_label", "input_len",
       "blocks_per_row", "topk_val"]

KERNEL_ORDER = ["topk_output_v2", "topk_output_v2_remap",
                "approx_topk_output", "approx_topk_output_remap"]
KERNEL_LABEL = {
    "topk_output_v2"          : r"$\mathtt{topk\_v2}$",
    "topk_output_v2_remap"    : r"$\mathtt{topk\_v2}$ + remap",
    "approx_topk_output"      : r"$\mathtt{approx\_topk}$",
    "approx_topk_output_remap": r"$\mathtt{approx\_topk}$ + remap",
}
# colorblind-safe palette (Okabe-Ito inspired)
KERNEL_COLOR = {
    "topk_output_v2"          : "#0072B2",   # blue
    "topk_output_v2_remap"    : "#56B4E9",   # light blue
    "approx_topk_output"      : "#D55E00",   # vermilion
    "approx_topk_output_remap": "#F0A270",   # light vermilion
}
BASELINE_COLOR = {
    "topk_output"           : "#555555",
    "topk_output_sglang_ori": "#999999",
}
MODEL_LABEL = {
    "qwen3_0p6b": "0.6B",
    "qwen3_1p7b": "1.7B",
    "qwen3_4b"  : "4B",
    "qwen3_8b"  : "8B",
}
DIST_LABEL = {
    "uniform": "uniform",
    "normal" : "normal",
    "bimodal": "bimodal",
    "real"   : "real",
}


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df["kernel_base"] = df["kernel"].str.replace(r"@.*$", "", regex=True)
    return df[df["input_label"] == "32k"].copy()


def make_pivot(df: pd.DataFrame) -> pd.DataFrame:
    best = (df.groupby(KEY + ["kernel_base"])["mean_ms"]
              .min().reset_index())
    return best.pivot_table(index=KEY, columns="kernel_base",
                            values="mean_ms").reset_index()


def add_speedups(pv: pd.DataFrame) -> pd.DataFrame:
    for k in KERNEL_ORDER:
        pv[f"spd_{k}_vs_full"] = pv["topk_output"] / pv[k]
        pv[f"spd_{k}_vs_sg"]   = pv["topk_output_sglang_ori"] / pv[k]
    return pv


# ============================================================================
# Single combined figure: 2 x 2 panels (speedup top, correctness bottom)
# ============================================================================
def fig_combined(df: pd.DataFrame, pv: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(17, 3.6))
    gs  = fig.add_gridspec(1, 4, width_ratios=[1.7, 0.9, 1.7, 1.3],
                           wspace=0.34)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[0, 3])

    # --- (a) speedup vs topk_output, by k --------------------------------
    topk_vals = sorted(pv["topk_val"].unique())
    width     = 0.19
    x         = np.arange(len(topk_vals))
    for i, k in enumerate(KERNEL_ORDER):
        means = [pv[pv["topk_val"] == kv][f"spd_{k}_vs_full"].mean()
                 for kv in topk_vals]
        stds  = [pv[pv["topk_val"] == kv][f"spd_{k}_vs_full"].std()
                 for kv in topk_vals]
        ax_a.bar(x + (i - 1.5) * width, means, width,
                 yerr=stds, capsize=2,
                 label=KERNEL_LABEL[k], color=KERNEL_COLOR[k],
                 edgecolor="black", linewidth=0.4)
    ax_a.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([f"$k$={kv}" for kv in topk_vals])
    ax_a.set_xlabel("Number of selected blocks")
    ax_a.set_ylabel(r"Speedup vs $\mathtt{topk\_output}$ ($\times$)")
    ax_a.set_title(r"(a) Speedup vs $\mathtt{topk\_output}$ (full-sort)")
    ax_a.set_ylim(0, 2.15)
    ax_a.grid(axis="y", lw=0.3, alpha=0.5)
    ax_a.legend(ncol=2, loc="upper right", framealpha=0.9,
                handlelength=1.6, columnspacing=1.0)

    # --- (b) speedup vs SGLang at k=32 -----------------------------------
    sub = pv[pv["topk_output_sglang_ori"].notna()]
    means = [sub[f"spd_{k}_vs_sg"].mean() for k in KERNEL_ORDER]
    stds  = [sub[f"spd_{k}_vs_sg"].std()  for k in KERNEL_ORDER]
    bx    = np.arange(len(KERNEL_ORDER))
    bars  = ax_b.bar(bx, means, yerr=stds, capsize=2,
                     color=[KERNEL_COLOR[k] for k in KERNEL_ORDER],
                     edgecolor="black", linewidth=0.4)
    for r, v in zip(bars, means):
        ax_b.text(r.get_x() + r.get_width() / 2, v + 0.06,
                  f"{v:.2f}$\\times$", ha="center", fontsize=8.5)
    ax_b.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax_b.set_xticks(bx)
    ax_b.set_xticklabels(["v2", "v2+R", "approx", "approx+R"], rotation=15)
    ax_b.set_ylabel(r"Speedup vs SGLang ($\times$)")
    ax_b.set_title(r"(b) vs $\mathtt{sglang\_ori}$, $k$=32")
    ax_b.set_ylim(0, max(means) * 1.35)
    ax_b.grid(axis="y", lw=0.3, alpha=0.5)

    # --- best variant per (config, kernel_base) for recall panels --------
    best_idx = df.groupby(KEY + ["kernel_base"])["mean_ms"].idxmin()
    best = df.loc[best_idx]

    # --- (c) recall by kernel and k --------------------------------------
    kernels  = ["topk_output", "topk_output_sglang_ori"] + KERNEL_ORDER
    palette  = {**KERNEL_COLOR, **BASELINE_COLOR}
    pretty   = {**KERNEL_LABEL,
                "topk_output"           : r"$\mathtt{topk\_output}$",
                "topk_output_sglang_ori": r"$\mathtt{sglang\_ori}$"}
    width = 0.13
    x     = np.arange(len(topk_vals))
    for i, k in enumerate(kernels):
        means = []
        for kv in topk_vals:
            s = best[(best["kernel_base"] == k) & (best["topk_val"] == kv)]
            means.append(s["recall_at_topk"].mean() if len(s) else np.nan)
        ax_c.bar(x + (i - len(kernels) / 2 + 0.5) * width, means, width,
                 label=pretty[k], color=palette[k],
                 edgecolor="black", linewidth=0.4)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([f"$k$={kv}" for kv in topk_vals])
    ax_c.set_xlabel("Number of selected blocks")
    ax_c.set_ylabel("Mean recall @ top-$k$")
    ax_c.set_ylim(0, 1.05)
    ax_c.set_title("(c) Recall (SGLang $\\approx$ 0)")
    ax_c.legend(ncol=2, loc="lower right", framealpha=0.9,
                handlelength=1.2, columnspacing=0.8, fontsize=7)
    ax_c.grid(axis="y", lw=0.3, alpha=0.5)

    # --- (d) latency vs recall scatter (zoomed Pareto) -------------------
    for k in KERNEL_ORDER:
        s = best[best["kernel_base"] == k]
        ax_d.scatter(s["mean_ms"] * 1000.0, s["recall_at_topk"],
                     s=28, alpha=0.85, color=KERNEL_COLOR[k],
                     edgecolor="black", linewidth=0.3,
                     label=KERNEL_LABEL[k])
    bsub = best[best["kernel_base"] == "topk_output"]
    ax_d.scatter(bsub["mean_ms"] * 1000.0, bsub["recall_at_topk"],
                 s=40, marker="X",
                 color=BASELINE_COLOR["topk_output"],
                 edgecolor="black", linewidth=0.3,
                 label=r"$\mathtt{topk\_output}$")
    ax_d.annotate("", xy=(22.5, 1.001), xytext=(33, 0.985),
                  arrowprops=dict(arrowstyle="->", color="gray", lw=1.0))
    ax_d.text(22.7, 1.002, "better", color="gray", fontsize=8.5)
    ax_d.set_xlabel(r"Latency ($\mu$s)")
    ax_d.set_ylabel("Recall @ top-$k$")
    ax_d.set_title("(d) Latency--recall (zoomed)")
    ax_d.set_ylim(0.974, 1.005)
    ax_d.grid(lw=0.3, alpha=0.5)
    ax_d.legend(loc="lower left", framealpha=0.9, handlelength=1.2,
                fontsize=7.5)

    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


# ============================================================================
def summary(pv: pd.DataFrame) -> None:
    print("\n=== speedup vs topk_output (full-sort) ===")
    for k in KERNEL_ORDER:
        s = pv[f"spd_{k}_vs_full"]
        print(f"{k:30s}  mean={s.mean():.2f}x  median={s.median():.2f}x  "
              f"min={s.min():.2f}x  max={s.max():.2f}x")
    print("\n=== speedup vs sglang_ori (k=32) ===")
    sub = pv[pv["topk_output_sglang_ori"].notna()]
    for k in KERNEL_ORDER:
        s = sub[f"spd_{k}_vs_sg"]
        print(f"{k:30s}  mean={s.mean():.2f}x  min={s.min():.2f}x  "
              f"max={s.max():.2f}x")
    print("\n=== mean speedup per topk_val ===")
    print(pv.groupby("topk_val")[
        [f"spd_{k}_vs_full" for k in KERNEL_ORDER]].mean().round(3))
    print("\n=== mean speedup per distribution ===")
    print(pv.groupby("distribution")[
        [f"spd_{k}_vs_full" for k in KERNEL_ORDER]].mean().round(3))


def main() -> None:
    df = load()
    pv = add_speedups(make_pivot(df))
    fig_combined(df, pv)
    summary(pv)


if __name__ == "__main__":
    main()
