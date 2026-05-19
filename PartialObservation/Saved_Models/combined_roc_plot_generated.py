import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from pathlib import Path
import re

# =========================
# CONFIG
# =========================

ROOT = Path("/mnt/110e4fe8-b416-4f34-8f8b-66a64d07a44f/anjani/partial/PartialObservation-main/final/PartialObservation/Saved_Models")

METHODS = [
    "decouple_cnn_mod/SNRs",
    "standalone_cnn/SNRs",
    "FL/SNRs/RandModTrTe",
    "EnergyDetect/RandMod"
]

SNRS = [0,-4, -8, -10 ]

OUTPUT = ROOT / "generated_roc.png"

# =========================
# STYLE SETTINGS
# =========================

METHOD_STYLES = {
    "decouple_cnn_mod/SNRs": {
        "label": "Proposed",
        "color": "red",
        "linestyle": "-"
    },

    "standalone_cnn/SNRs": {
        "label": "Standalone",
        "color": "deepskyblue",
        "linestyle": "--"
    },

    "FL/SNRs/RandModTrTe": {
        "label": "FL",
        "color": "limegreen",
        "linestyle": "-."
    },

    "EnergyDetect/RandMod": {
        "label": "ED",
        "color": "gray",
        "linestyle": ":"
    }
}

# =========================
# READ XLSX
# =========================

def read_roc_xlsx(path):

    df = pd.read_excel(path)

    pfa_col = next(
        (c for c in df.columns if str(c).strip().lower() == "pfa"),
        None
    )

    pd_col = next(
        (c for c in df.columns if str(c).strip().lower() == "pd"),
        None
    )

    pfa = pd.to_numeric(df[pfa_col], errors="coerce")
    pd_vals = pd.to_numeric(df[pd_col], errors="coerce")

    valid = pfa.notna() & pd_vals.notna()

    return (
        pfa[valid].astype(float).tolist(),
        pd_vals[valid].astype(float).tolist()
    )

# =========================
# EXTRACT SNR
# =========================

def snr_from_path(path):

    match = re.search(r"ROC_SNR(-?\d+)", path.name)

    if match:
        return int(match.group(1))

    return None

# =========================
# MAIN PLOT
# =========================

def main():

    plt.figure(figsize=(6.5, 6), dpi=180)

    legend_added = set()

    for method in METHODS:

        style = METHOD_STYLES[method]

        for snr in SNRS:

            method_path = ROOT / method / "GeneratedDataset"

            files = [
                p for p in method_path.rglob(f"ROC_SNR{snr}.xlsx")
                if snr_from_path(p) == snr
            ]

            if not files:
                continue

            roc_file = max(files, key=lambda p: p.stat().st_mtime)

            try:
                pfa, pd_vals = read_roc_xlsx(roc_file)

            except Exception as e:
                print(f"Error reading {roc_file}: {e}")
                continue

            # convert to percentage if needed
            if max(pfa) <= 1.5:
                pfa = [x * 100 for x in pfa]

            if max(pd_vals) <= 1.5:
                pd_vals = [x * 100 for x in pd_vals]

            # sort points
            paired = sorted(zip(pfa, pd_vals))
            pfa, pd_vals = zip(*paired)

            # add legend only once
            label = None

            if method not in legend_added:
                label = style["label"]
                legend_added.add(method)

            plt.plot(
                pfa,
                pd_vals,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.8,
                label=label
            )

    # =========================
    # AXIS STYLE
    # =========================

    plt.xlim(0, 100)
    plt.ylim(0, 100)

    plt.xlabel(
        "Probability of false alarm (%)",
        fontsize=10,
        fontweight="bold",
        family="serif"
    )

    plt.ylabel(
        "Probability of detection (%)",
        fontsize=10,
        fontweight="bold",
        family="serif"
    )

    plt.xticks(
        range(0, 101, 10),
        family="serif"
    )

    plt.yticks(
        range(0, 101, 10),
        family="serif"
    )

    # grid
    plt.grid(
        True,
        linestyle="--",
        linewidth=0.8,
        alpha=0.6
    )

    # thicker borders
    ax = plt.gca()

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    # =========================
    # SNR GROUP MARKERS
    # =========================

    # ---------- -10 dB ----------
    ellipse1 = Ellipse(
        xy=(35, 90),       # center
        width=8,
        height=12,
        angle=0,
        edgecolor='darkorange',
        facecolor='none',
        linewidth=1.5
    )

    ax.add_patch(ellipse1)

    plt.text(
        39,
        89,
        "-10 dB",
        fontsize=10,
        fontweight='bold',
        family='serif',
        bbox=dict(
            facecolor='white',
            edgecolor='none',
            pad=0.2
        )
    )

    # ---------- -16 dB ----------
    ellipse2 = Ellipse(
        xy=(67, 80),
        width=8,
        height=14,
        angle=0,
        edgecolor='darkorange',
        facecolor='none',
        linewidth=1.5
    )

    ax.add_patch(ellipse2)

    plt.text(
        71,
        78,
        "-16 dB",
        fontsize=10,
        fontweight='bold',
        family='serif',
        bbox=dict(
            facecolor='white',
            edgecolor='none',
            pad=0.2
        )
    )

    # =========================
    # LEGEND
    # =========================

    plt.legend(
        loc="lower right",
        fontsize=9,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=1,
        handlelength=4,
        borderpad=0.8,
        prop={
            "family": "serif",
            "weight": "bold"
        }
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT,
        bbox_inches="tight",
        facecolor="white"
    )

    print(f"\nSaved paper-style ROC plot to:\n{OUTPUT}")

# =========================

if __name__ == "__main__":
    main()