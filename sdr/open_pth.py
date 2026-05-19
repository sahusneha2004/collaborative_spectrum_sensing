import csv
import os
from pathlib import Path

import numpy as np
import torch
from torch.serialization import safe_globals

path = "/mnt/110e4fe8-b416-4f34-8f8b-66a64d07a44f/anjani/data/sdr/GeneratedDatasets_realistic/260103_001533/Data_SNR-2vol20.pth"
SAVE_N = 10
output_dir = Path(path).parent
output_dir.mkdir(parents=True, exist_ok=True)


def load_pth(path):
    with safe_globals([np._core.multiarray._reconstruct]):
        return torch.load(path, map_location="cpu", weights_only=False)


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)):
        return np.array(value)
    return np.array([value])


def save_csv(file_path, rows):
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)


def save_list_items(name, items, n=SAVE_N):
    if not isinstance(items, list):
        return

    rows = []
    for item in items[:n]:
        if isinstance(item, (torch.Tensor, np.ndarray)):
            arr = to_numpy(item).ravel()
            rows.append(arr.tolist())
        elif isinstance(item, (list, tuple)):
            rows.append(to_numpy(item).ravel().tolist())
        else:
            rows.append([item])

    if rows:
        output_path = output_dir / f"{name.replace(' ', '_')}_top{len(rows)}.csv"
        save_csv(output_path, rows)
        print(f"Saved {len(rows)} rows to {output_path}")


def save_dict_items(name, value, n=SAVE_N):
    if isinstance(value, dict):
        summary_rows = [["key", "type", "shape", "sample"]]
        for k, v in list(value.items())[:n]:
            sample = ""
            if isinstance(v, (torch.Tensor, np.ndarray)):
                sample = str(to_numpy(v).ravel()[:5].tolist())
            elif isinstance(v, list):
                sample = str(v[:5])
            else:
                sample = str(v)
            summary_rows.append([k, type(v).__name__, getattr(v, "shape", ""), sample])
        output_path = output_dir / f"{name.replace(' ', '_')}_summary.csv"
        save_csv(output_path, summary_rows)
        print(f"Saved dict summary to {output_path}")


obj = load_pth(path)
print("Loaded object type:", type(obj))

if isinstance(obj, dict):
    print("Keys:", list(obj.keys()))
    for key, value in obj.items():
        if isinstance(value, list):
            save_list_items(key, value)
        elif isinstance(value, (torch.Tensor, np.ndarray)):
            save_list_items(key, [value], n=1)
        elif isinstance(value, dict):
            save_dict_items(key, value)
        else:
            print(f"Skipping key {key}: type {type(value).__name__}")
else:
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        save_list_items("data", [obj], n=1)
    elif isinstance(obj, list):
        save_list_items("data", obj)
    else:
        print("Top-level object is not list/dict/tensor; skipping CSV export.")
