import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("/mnt/110e4fe8-b416-4f34-8f8b-66a64d07a44f/anjani/partial/PartialObservation-main/final/PartialObservation/Saved_Models/standalone_cnn/SNRs/GeneratedDataset") 
SNRS = [0 , -2, -4, -6, -8, -10, -12, -14, -16]  # SNRs to plot
OUTPUT = Path("/mnt/110e4fe8-b416-4f34-8f8b-66a64d07a44f/anjani/partial/PartialObservation-main/final/PartialObservation/Saved_Models/standalone_cnn/SNRs/GeneratedDataset/standalone_cnn_roc.png")


def read_roc_csv(path):
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    pfa = []
    pd_values = []
    for row in rows:
        if row.get("PFA") in ("", None) or row.get("PD") in ("", None):
            continue
        pfa.append(float(row["PFA"]))
        pd_values.append(float(row["PD"]))
    return pfa, pd_values


def read_roc_xlsx(path):
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("Reading .xlsx ROC files needs pandas/openpyxl installed.") from exc

    df = pd.read_excel(path)
    pfa_col = next((col for col in df.columns if str(col).strip().lower() == "pfa"), None)
    pd_col = next((col for col in df.columns if str(col).strip().lower() == "pd"), None)

    if pfa_col is None or pd_col is None:
        raise ValueError(f"{path} does not contain PFA and PD columns.")

    pfa = pd.to_numeric(df[pfa_col], errors="coerce")
    pd_values = pd.to_numeric(df[pd_col], errors="coerce")
    valid = pfa.notna() & pd_values.notna()
    return pfa[valid].astype(float).tolist(), pd_values[valid].astype(float).tolist()


def read_roc_file(path):
    if path.suffix.lower() == ".csv":
        return read_roc_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return read_roc_xlsx(path)
    raise ValueError(f"Unsupported ROC file type: {path}")


def snr_from_path(path):
    match = re.search(r"ROC_SNR(-?\d+)", path.name, re.IGNORECASE)
    if match:
        return int(match.group(1))

    match = re.search(r"(-?\d+)dB", str(path))
    if match:
        return int(match.group(1))

    return None


def main():
    plt.figure(figsize=(8, 6), dpi=140)
    plotted = 0
    all_values = []

    for snr in SNRS:
        files = [
            p
            for p in list(ROOT.rglob(f"ROC_SNR{snr}.csv")) + list(ROOT.rglob(f"ROC_SNR{snr}.xlsx"))
            if snr_from_path(p) == snr
        ]

        if not files:
            print(f"Missing standalone_cnn ROC for SNR {snr}; skipping.")
            continue

        # If there are multiple runs, use the newest result file.
        roc_file = max(files, key=lambda p: p.stat().st_mtime)
        pfa, pd_values = read_roc_file(roc_file)

        if not pfa or not pd_values:
            print(f"Found {roc_file}, but it has no numeric PFA/PD rows; skipping.")
            continue

        print(
            f"SNR {snr}: {len(pfa)} points, "
            f"PFA range {min(pfa):.4g}..{max(pfa):.4g}, "
            f"PD range {min(pd_values):.4g}..{max(pd_values):.4g}"
        )

        plt.plot(pfa, pd_values, marker="o", markersize=3, linewidth=1.8, label=f"SNR {snr} dB")
        print(f"Plotted SNR {snr} from {roc_file}")
        plotted += 1
        all_values.extend(pfa)
        all_values.extend(pd_values)

    if plotted == 0:
        print("No standalone_cnn ROC files found. Nothing plotted.")
        return

    plt.xlabel("Probability of False Alarm (PFA)")
    plt.ylabel("Probability of Detection (PD)")
    plt.title("Standalone CNN ROC Curves")
    plt.grid(True, alpha=0.3)

    if max(all_values) <= 1.5:
        plt.xlim(0, 1)
        plt.ylim(0, 1)
    else:
        plt.xlim(0, 100)
        plt.ylim(0, 105)

    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT)
    print(f"Saved plot to {OUTPUT}")


if __name__ == "__main__":
    main()
