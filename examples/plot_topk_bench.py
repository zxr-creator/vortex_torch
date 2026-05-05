"""Plot top-k kernel latency for a top-ML-conference figure.

A 2 x 4 grid of subfigures, single PDF:
  Row 1 (top)    : short context  (2k)   — one column per distribution
  Row 2 (bottom) : long context   (128k) — one column per distribution

Distributions: bimodal / normal / real / uniform.
Each subfigure: grouped bars where x = model size and groups = sparse
top-k algorithm family. Per-config best variant (min mean_ms across
mapping / tolerate-ratio variants). No error bars.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV = Path("/data/datasets/xinrui/My_Projects/v0.3/vortex_torch/examples/topk_bench_20260505_215943.csv")
OUT = Path("/data/datasets/xinrui/topk_kernels.pdf")

LOW_LABEL  = "2k"
LONG_LABEL = "128k"

# ----- conference-quality matplotlib defaults --------------------------------
mpl.rcParams.update({
    "font.family"      : "serif",
    "font.serif"       : ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size"        : 9.0,
    "axes.titlesize"   : 10,
    "axes.labelsize"   : 9.5,
    "xtick.labelsize"  : 8.5,
    "ytick.labelsize"  : 8.5,
    "legend.fontsize"  : 8.8,
    "axes.linewidth"   : 0.8,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "pdf.fonttype"     : 42,
    "ps.fonttype"      : 42,
})


# ----- kernel family taxonomy ------------------------------------------------
def family_of(kernel: str) -> str:
    if kernel == "sort_topk":
        return "sort_topk"
    if kernel.startswith("approx_radix_topk_remap"):
        return "approx_radix_topk_remap"
    if kernel.startswith("approx_radix_topk"):
        return "approx_radix_topk"
    if kernel.startswith("radix_topk_remap"):
        return "radix_topk_remap"
    if kernel == "radix_topk":
        return "radix_topk"
    return "other"


FAMILY_ORDER = [
    "radix_topk",
    "radix_topk_remap",
    "approx_radix_topk",
    "approx_radix_topk_remap",
]
FAMILY_LABEL = {
    "radix_topk"              : r"$\mathtt{radix\_topk}$",
    "radix_topk_remap"        : r"$\mathtt{radix\_topk}$ + remap",
    "approx_radix_topk"       : r"$\mathtt{approx\_radix\_topk}$",
    "approx_radix_topk_remap" : r"$\mathtt{approx\_radix\_topk}$ + remap",
}

# Soft pastel palette inspired by the supplied reference figure.
FAMILY_COLOR = {
    "radix_topk"              : "#F2B07A",  # soft peach
    "radix_topk_remap"        : "#E89AAE",  # blush pink
    "approx_radix_topk"       : "#9CC6A8",  # sage green
    "approx_radix_topk_remap" : "#C5A6D6",  # soft lavender
}

MODEL_ORDER = ["qwen3_0p6b", "qwen3_1p7b", "qwen3_4b", "qwen3_8b"]
MODEL_LABEL = {
    "qwen3_0p6b": "0.6B",
    "qwen3_1p7b": "1.7B",
    "qwen3_4b"  : "4B",
    "qwen3_8b"  : "8B",
}

DIST_ORDER = ["bimodal", "normal", "real", "uniform"]
DIST_LABEL = {
    "bimodal": "bimodal",
    "normal" : "normal",
    "real"   : "real",
    "uniform": "uniform",
}


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df["family"] = df["kernel"].map(family_of)
    return df[df["input_label"].isin([LOW_LABEL, LONG_LABEL])].copy()


def latency_table(df: pd.DataFrame, input_label: str,
                  distribution: str) -> pd.DataFrame:
    """(model x family) latency table in ms for one (input_label, distribution).

    Per (model, family) we take the best (min mean_ms) across mapping /
    tolerate-ratio variants — no averaging across distributions.
    """
    sub = df[(df["input_label"] == input_label)
             & (df["distribution"] == distribution)]
    best = (sub.groupby(["model", "family"])["mean_ms"]
               .min()
               .reset_index())
    pv = best.pivot(index="model", columns="family", values="mean_ms")
    return pv.reindex(index=MODEL_ORDER, columns=FAMILY_ORDER)


def draw_panel(ax, pv: pd.DataFrame, title: str, ylabel_show: bool,
               xlabel_show: bool) -> None:
    n_fam   = len(FAMILY_ORDER)
    n_model = len(MODEL_ORDER)
    width   = 0.84 / n_fam
    x       = np.arange(n_model)

    for i, fam in enumerate(FAMILY_ORDER):
        vals = pv[fam].values
        offset = (i - (n_fam - 1) / 2) * width
        ax.bar(x + offset, vals, width,
               color=FAMILY_COLOR[fam],
               edgecolor="#404040", linewidth=0.5,
               label=FAMILY_LABEL[fam])

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL[m] for m in MODEL_ORDER])
    if xlabel_show:
        ax.set_xlabel("Model size")
    if ylabel_show:
        ax.set_ylabel("Latency (ms)")
    ax.set_title(title)
    ax.grid(axis="y", lw=0.3, alpha=0.55)
    ax.set_axisbelow(True)
    ymax = np.nanmax(pv.values)
    ax.set_ylim(0, ymax * 1.18)


def fig_grid(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 5.6),
                             gridspec_kw={"wspace": 0.30, "hspace": 0.42})

    rows = [(LOW_LABEL,  "Short context (2k)"),
            (LONG_LABEL, "Long context (128k)")]

    for r, (input_label, ctx_name) in enumerate(rows):
        for c, dist in enumerate(DIST_ORDER):
            ax = axes[r, c]
            pv = latency_table(df, input_label, dist)
            tag = chr(ord('a') + r * 4 + c)  # a..h
            title = f"({tag}) {ctx_name} – {DIST_LABEL[dist]}"
            draw_panel(ax, pv, title,
                       ylabel_show=(c == 0),
                       xlabel_show=(r == 1))

    # Single shared legend at the top.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=len(FAMILY_ORDER),
               frameon=False, handlelength=1.6, columnspacing=1.4)

    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")

    # Dump the underlying numbers for the paper text.
    for input_label, ctx_name in rows:
        for dist in DIST_ORDER:
            pv = latency_table(df, input_label, dist)
            print(f"\n=== latency (ms), {ctx_name}, distribution={dist} ===")
            print(pv.round(4))


def main() -> None:
    df = load()
    fig_grid(df)


if __name__ == "__main__":
    main()
