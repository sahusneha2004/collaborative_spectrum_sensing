import os
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

# ========================
# AUTO FIND DATA DIRECTORY
# ========================
def find_data_dir(base_path="/home/anjani/partial/sdr"):
    for root, dirs, files in os.walk(base_path):
        if any(f.startswith("Data_SNR") and f.endswith(".pth") for f in files):
            print(f"Found dataset in: {root}")
            return root
    raise FileNotFoundError("No dataset found. Run generator first.")

DATA_DIR = find_data_dir()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ========================
# CONFIG
# ========================
NUM_CHANNELS = 20
NW = 64
SEQ_LEN = 5
BATCH_SIZE = 32
EPOCHS = 15
TEST_SNR = -10
TARGET_PFA = 0.1
THRESHOLDS = np.linspace(0, 1, 300)

TIME_TAG = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
RESULTS_DIR = f"./final_2D_comparison_{TIME_TAG}"
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"Results saved to: {RESULTS_DIR}")

# ========================
# LOAD FILES
# ========================
def load_data_dir(data_dir):
    snr_files = {}
    for f in os.listdir(data_dir):
        if f.startswith("Data_SNR") and f.endswith(".pth"):
            fullname = os.path.join(data_dir, f)
            name_part = f[len("Data_SNR"):].split("vol")[0].lower()
            name_part = name_part.replace("m", "-").replace("p", ".")
            try:
                snr = float(name_part)
                snr_files[snr] = fullname
            except:
                continue
    return snr_files

snr_files = load_data_dir(DATA_DIR)
sorted_snrs = sorted(snr_files.keys())
print("Available SNRs:", sorted_snrs)

train_snr = max([s for s in sorted_snrs if s <= 0], default=sorted_snrs[0])
print(f"Training on SNR = {train_snr} dB")

# ========================
# 2D PSD DATA (Average over SUs)
# ========================
def get_2d_psd_data(data_dict):
    samples = data_dict['training data list']
    labels = torch.stack(data_dict['training label list'])

    psd_list = []
    for sample in samples:
        stacked = torch.stack(sample)      # (num_SU, 1, 64, 20)
        avg_psd = torch.mean(stacked, dim=0)  # (1, 64, 20)
        psd_list.append(avg_psd)

    X = torch.stack(psd_list)  # (N, 1, 64, 20)
    Y = labels.numpy()
    return X, Y

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

# ========================
# MODEL 1: Decoupled CNN (IEEE-style) — FIXED
# ========================
class DecoupleCNN(nn.Module):
    def __init__(self, nch=NUM_CHANNELS):
        super().__init__()
        self.nch = nch
        cfg = [40, 8*nch, 8*nch, 8*nch]  # 160 channels

        self.features = nn.Sequential(
            nn.Conv2d(1, cfg[0], 3, padding=1),
            nn.BatchNorm2d(cfg[0]),
            nn.ReLU(),

            nn.Conv2d(cfg[0], cfg[1], 3, padding=1),
            nn.BatchNorm2d(cfg[1]),
            nn.ReLU(),

            nn.MaxPool2d((4,1)),

            nn.Conv2d(cfg[1], cfg[2], 3, groups=nch, padding=1),
            nn.BatchNorm2d(cfg[2]),
            nn.ReLU(),

            nn.MaxPool2d((2,1)),

            nn.Conv2d(cfg[2], cfg[3], 3, groups=nch, padding=1),
            nn.BatchNorm2d(cfg[3]),
            nn.ReLU(),

            # ✅ CRITICAL FIX
            nn.AdaptiveAvgPool2d((1,1))
        )

        self.fc = nn.ModuleList([nn.Linear(8, 1) for _ in range(nch)])

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)

        x = self.features(x)              # (B*T, 160, 1, 1)
        x = x.squeeze(-1).squeeze(-1)     # (B*T, 160)
        x = x.view(B*T, self.nch, 8)      # (B*T, 20, 8)

        outs = [fc(x[:, i]) for i, fc in enumerate(self.fc)]
        out = torch.cat(outs, dim=1)      # (B*T, 20)

        return out.view(B, T, self.nch)[:, -1, :]

# ========================
# MODEL 2: Proposed 2D CNN + GRU + Attention
# ========================
class Your2DModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.c3 = nn.Conv2d(1, 32, 3, padding=1)
        self.c5 = nn.Conv2d(1, 32, 5, padding=2)
        self.c7 = nn.Conv2d(1, 32, 7, padding=3)

        self.relu = nn.ReLU(inplace=True)

        self.gru = nn.GRU(96, 128, batch_first=True)
        self.attn = nn.MultiheadAttention(128, 8, batch_first=True)
        self.fc = nn.Linear(128, NUM_CHANNELS)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)

        x3 = self.relu(self.c3(x))
        x5 = self.relu(self.c5(x))
        x7 = self.relu(self.c7(x))

        x = torch.cat([x3, x5, x7], dim=1)  # (B*T, 96, H, W)
        x = x.mean(dim=[2,3])               # (B*T, 96)

        x = x.view(B, T, 96)
        g, _ = self.gru(x)
        a, _ = self.attn(g, g, g)

        return self.fc(a[:, -1, :])

# ========================
# METRICS
# ========================
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

# ========================
# TRAIN + EVALUATE
# ========================
def train_and_evaluate():
    train_data = torch.load(snr_files[train_snr], map_location='cpu')
    X_train, Y_train = get_2d_psd_data(train_data)

    train_loader = DataLoader(
        PSD2DDataset(X_train, Y_train),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    models = {
        "Decoupled CNN (IEEE)": DecoupleCNN().to(DEVICE),
        "Proposed 2D CNN+GRU+Attn": Your2DModel().to(DEVICE)
    }

    criterion = nn.BCEWithLogitsLoss()

    print("\n=== TRAINING ===")
    for name, model in models.items():
        opt = optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for _ in range(EPOCHS):
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                opt.step()
        print(f"{name} trained")

    # ================= ROC =================
    if TEST_SNR in snr_files:
        plt.figure(figsize=(10,8))
        test_data = torch.load(snr_files[TEST_SNR], map_location='cpu')
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
        plt.title(f"ROC on Real SDR Data (SNR={TEST_SNR} dB)")
        plt.grid()
        plt.legend()
        plt.savefig(f"{RESULTS_DIR}/ROC_RealData.png", dpi=300)
        plt.close()

    # ================= Sensing Error =================
    results = []
    for snr in sorted_snrs:
        data = torch.load(snr_files[snr], map_location='cpu')
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
    df.to_csv(f"{RESULTS_DIR}/final_results_real_data.csv", index=False)

    plt.figure(figsize=(12,8))
    for name in df.Model.unique():
        sub = df[df.Model == name]
        plt.plot(sub.SNR, sub.Sensing_Error, 'o-', linewidth=3, label=name)

    plt.xlabel("SNR (dB)")
    plt.ylabel("Sensing Error (Pfa + Pmd)")
    plt.title("Performance on Real Wideband SDR Data")
    plt.grid()
    plt.legend()
    plt.savefig(f"{RESULTS_DIR}/SensingError_vs_SNR.png", dpi=300)
    plt.close()

    print("\n✅ SUCCESS")
    print("Saved:")
    print(" - ROC_RealData.png")
    print(" - SensingError_vs_SNR.png")
    print(" - final_results_real_data.csv")

# ========================
if __name__ == "__main__":
    train_and_evaluate()
