import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re

# =========================
# CONFIG
# =========================

ROOT = Path("/mnt/110e4fe8-b416-4f34-8f8b-66a64d07a44f/anjani/partial/PartialObservation-main/final/PartialObservation/Saved_Models")

METHODS = [
    "decouple_cnn_mod/SNRs/puswitch",
    "Transformer/PUswitch"
]

SNRS = [-8]

OUTPUT = ROOT / "decoupled_cnn_vs_transformer_puswitch.png"

# =========================
# STYLE SETTINGS
# =========================

METHOD_STYLES = {

    "decouple_cnn_mod/SNRs/puswitch": {
        "label": "Decoupled CNN",
        "color": "red",
        "linestyle": "-"
    },

    "Transformer/PUswitch": {
        "label": "Transformer",
        "color": "blue",
        "linestyle": "--",
        "label_color": "black"
    }
}

# =========================
# READ XLSX
# =========================

def read_roc_xlsx(path):

    df = pd.read_excel(path)

    pfa_col = next(
        (c for c in df.columns
         if str(c).strip().lower() == "pfa"),
        None
    )

    pd_col = next(
        (c for c in df.columns
         if str(c).strip().lower() == "pd"),
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

            method_path = ROOT / method

            files = [
                p for p in method_path.rglob(f"ROC_SNR{snr}.xlsx")
                if snr_from_path(p) == snr
            ]

            if not files:
                print(f"No file found for {method}, SNR={snr}")
                continue

            roc_file = max(
                files,
                key=lambda p: p.stat().st_mtime
            )

            try:
                pfa, pd_vals = read_roc_xlsx(roc_file)

            except Exception as e:
                print(f"Error reading {roc_file}: {e}")
                continue

            # =========================
            # CONVERT TO %
            # =========================

            if max(pfa) <= 1.5:
                pfa = [x * 100 for x in pfa]

            if max(pd_vals) <= 1.5:
                pd_vals = [x * 100 for x in pd_vals]

            # =========================
            # SORT POINTS
            # =========================

            paired = sorted(zip(pfa, pd_vals))
            pfa, pd_vals = zip(*paired)

            # =========================
            # LEGEND LABEL ONLY ONCE
            # =========================

            label = None

            if method not in legend_added:
                label = style["label"]
                legend_added.add(method)

            # =========================
            # PLOT CURVE
            # =========================

            plt.plot(
                pfa,
                pd_vals,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.0,
                label=label
            )

            # =========================
            # ADD INLINE SNR LABELS
            # ONLY FOR TRANSFORMER
            # =========================

            if method == "Transformer/PUswitch":

                # choose label point
                idx = int(len(pfa) * 0.65)

                # slight offset
                x_pos = pfa[idx] + 2
                y_pos = pd_vals[idx] - 2

                plt.text(
                    x_pos,
                    y_pos,
                    f"{snr} dB",

                    fontsize=9,
                    fontweight="bold",
                    family="serif",
                    color=style["label_color"],

                    bbox=dict(
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.7,
                        pad=0.2
                    )
                )

    # =========================
    # AXIS STYLE
    # =========================

    plt.xlim(0, 100)
    plt.ylim(0, 100)

    plt.xlabel(
        "Probability of false alarm (%)",
        fontsize=11,
        fontweight="bold",
        family="serif"
    )

    plt.ylabel(
        "Probability of detection (%)",
        fontsize=11,
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

    # =========================
    # GRID
    # =========================

    plt.grid(
        True,
        linestyle="--",
        linewidth=0.8,
        alpha=0.6
    )

    # =========================
    # THICKER BORDER
    # =========================

    ax = plt.gca()

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

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

    # =========================
    # SAVE
    # =========================

    plt.tight_layout()

    plt.savefig(
        OUTPUT,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.show()

    print(f"\nSaved plot to:\n{OUTPUT}")

# =========================

if __name__ == "__main__":
    main()