import os
import re
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# ROBUST DATASET FINDER
# ============================================================
# def find_data_dir(base_path="/home/anjani/partial/sdr"):
def find_data_dir(base_path="/mnt/110e4fe8-b416-4f34-8f8b-66a64d07a44f/anjani/partial/sdr"):
    for root, _, files in os.walk(base_path):
        for f in files:
            if f.lower().endswith(".pth") and "snr" in f.lower():
                print(f"Found dataset in: {root}")
                return root
    raise FileNotFoundError(
        "Dataset not found. Expected .pth files containing 'SNR' in filename."
    )

DATA_DIR = find_data_dir()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ============================================================
# CONFIG
# ============================================================
NUM_CHANNELS = 20
SEQ_LEN = 5
BATCH_SIZE = 32
EPOCHS = 30
TARGET_PFA = 0.1
DESIRED_TEST_SNR = -10
THRESHOLDS = np.linspace(0, 1, 300)

TIME_TAG = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
RESULTS_DIR = os.path.join(DATA_DIR, f"Results_3Model_Comparison_{TIME_TAG}")
os.makedirs(RESULTS_DIR, exist_ok=True)
print("Results saved to:", RESULTS_DIR)

# ============================================================
# LOAD DATA FILES (ROBUST SNR PARSING)
# ============================================================
def load_data_dir(data_dir):
    snr_files = {}
    for f in os.listdir(data_dir):
        if f.lower().endswith(".pth") and "snr" in f.lower():
            m = re.search(r"snr[_\-]?\s*(-?\d+)", f.lower())
            if m:
                snr = float(m.group(1))
                snr_files[snr] = os.path.join(data_dir, f)
    if len(snr_files) == 0:
        raise RuntimeError("SNR files found, but SNR parsing failed.")
    return snr_files

snr_files = load_data_dir(DATA_DIR)
sorted_snrs = sorted(snr_files.keys())
print("Available SNRs:", sorted_snrs)

train_snr = max([s for s in sorted_snrs if s <= 0], default=sorted_snrs[0])
TEST_SNR = min(sorted_snrs, key=lambda x: abs(x - DESIRED_TEST_SNR))

print(f"Training SNR: {train_snr} dB")
print(f"ROC will be generated at SNR = {TEST_SNR} dB")

# ============================================================
# DATASET
# ============================================================
def get_2d_psd_data(data_dict):
    samples = data_dict["training data list"]
    labels = torch.stack(data_dict["training label list"])

    psd_list = []
    for sample in samples:
        avg_psd = torch.mean(torch.stack(sample), dim=0)
        psd_list.append(avg_psd)

    return torch.stack(psd_list), labels.numpy()

class PSD2DDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X) - SEQ_LEN

    def __getitem__(self, idx):
        return (
            self.X[idx:idx + SEQ_LEN],
            torch.tensor(self.Y[idx + SEQ_LEN], dtype=torch.float32)
        )

# ============================================================
# MODEL 1: DECOUPLED CNN
# ============================================================
class DecoupleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 40, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(40, 160, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(160, NUM_CHANNELS)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        x = self.features(x).squeeze(-1).squeeze(-1)
        x = self.fc(x)
        return x.view(B, T, -1)[:, -1, :]

# ============================================================
# MODEL 2: CNN + GRU
# ============================================================
class CNN_GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 64, 3, padding=1)
        self.gru = nn.GRU(64, 128, batch_first=True)
        self.fc = nn.Linear(128, NUM_CHANNELS)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        x = torch.relu(self.conv(x)).mean(dim=[2, 3])
        x = x.view(B, T, -1)
        g, _ = self.gru(x)
        return self.fc(g[:, -1, :])
# ============================================================
# MODEL 3: HAPPO-INSPIRED CENTRALIZED POLICY
# ============================================================
class HAPPO_Model(nn.Module):
    """
    Offline HAPPO-inspired centralized actor
    (centralized state, decentralized band outputs)
    """
    def __init__(self):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(64 * 20, 256),
            nn.ReLU(),
            nn.Linear(256, NUM_CHANNELS)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        state = x[:, -1].reshape(B, -1)
        return self.policy(state)

# ============================================================
# METRICS
# ============================================================
def compute_pd_pfa(y_true, y_score):
    Pd, Pfa = [], []
    for th in THRESHOLDS:
        y_pred = (y_score >= th).astype(int)
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FN = np.sum((y_pred == 0) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        TN = np.sum((y_pred == 0) & (y_true == 0))
        Pd.append(TP / (TP + FN + 1e-12))
        Pfa.append(FP / (FP + TN + 1e-12))
    return np.array(Pd), np.array(Pfa)

# ============================================================
# TRAIN & EVALUATE
# ============================================================
def train_and_evaluate():
    train_data = torch.load(snr_files[train_snr], map_location="cpu")
    X_train, Y_train = get_2d_psd_data(train_data)

    train_loader = DataLoader(
        PSD2DDataset(X_train, Y_train),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    models = {
        "Decoupled CNN": DecoupleCNN().to(DEVICE),
        "CNN + GRU Proposed": CNN_GRU().to(DEVICE),
        "HAPPO-inspired": HAPPO_Model().to(DEVICE)
    }

    criterion = nn.BCEWithLogitsLoss()

    print("\n=== TRAINING ===")
    for name, model in models.items():
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for _ in range(EPOCHS):
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
        print(f"{name} trained")

    # ================= ROC =================
    plt.figure(figsize=(10, 8))
    test_data = torch.load(snr_files[TEST_SNR], map_location="cpu")
    X_test, Y_test = get_2d_psd_data(test_data)
    loader = DataLoader(PSD2DDataset(X_test, Y_test), batch_size=BATCH_SIZE)

    for name, model in models.items():
        model.eval()
        scores = []
        with torch.no_grad():
            for xb, _ in loader:
                scores.append(torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy())
        scores = np.concatenate(scores).flatten()
        truth = Y_test[SEQ_LEN:].flatten()
        Pd, Pfa = compute_pd_pfa(truth, scores)
        plt.plot(Pfa, Pd, linewidth=3, label=name)

    plt.xlabel("Pfa")
    plt.ylabel("Pd")
    plt.title(f"ROC Comparison (SNR = {TEST_SNR} dB)")
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "ROC_Comparison.pdf"), dpi=300)
    plt.close()

    # ================= SENSING ERROR =================
    results = []
    plt.figure(figsize=(12, 8))

    for snr in sorted_snrs:
        data = torch.load(snr_files[snr], map_location="cpu")
        X, Y = get_2d_psd_data(data)
        loader = DataLoader(PSD2DDataset(X, Y), batch_size=BATCH_SIZE)

        for name, model in models.items():
            model.eval()
            scores = []
            with torch.no_grad():
                for xb, _ in loader:
                    scores.append(torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy())
            scores = np.concatenate(scores).flatten()
            truth = Y[SEQ_LEN:].flatten()

            th = np.quantile(scores[truth == 0], 1 - TARGET_PFA)
            Pd = np.mean(scores[truth == 1] >= th)
            error = TARGET_PFA + (1 - Pd)

            results.append({"Model": name, "SNR": snr, "Sensing_Error": error})

    df = pd.DataFrame(results)

    for name in df.Model.unique():
        sub = df[df.Model == name]
        plt.plot(sub.SNR, sub.Sensing_Error, 'o-', linewidth=3, label=name)

    plt.xlabel("SNR (dB)")
    plt.ylabel("Sensing Error (Pfa + Pmd)")
    plt.title("Sensing Error vs SNR (3-Model Comparison)")
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "SensingError_vs_SNR.pdf"), dpi=300)
    plt.close()

    df.to_csv(os.path.join(RESULTS_DIR, "final_results_real_data.csv"), index=False)

    print("\n✅ SUCCESS")
    print("Saved files:")
    print(" - ROC_Comparison.pdf")
    print(" - SensingError_vs_SNR.pdf")
    print(" - final_results_real_data.csv")
    print("Location:", RESULTS_DIR)

# ============================================================
if __name__ == "__main__":
    train_and_evaluate()