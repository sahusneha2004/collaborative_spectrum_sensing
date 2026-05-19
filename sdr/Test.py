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
# DATASET FINDER
# ============================================================
def find_data_dir(base_path="/home/anjani/partial/sdr"):
    for root, _, files in os.walk(base_path):
        for f in files:
            if f.lower().endswith(".pth") and "snr" in f.lower():
                return root
    raise FileNotFoundError("Dataset not found")

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

TIME_TAG = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = os.path.join(DATA_DIR, f"Results_3Model_{TIME_TAG}")
AUC_DIR = os.path.join(RESULTS_DIR, "AUC_vs_SNR")
ROC_CH_DIR = os.path.join(RESULTS_DIR, "ROC_Per_Channel")

os.makedirs(AUC_DIR, exist_ok=True)
os.makedirs(ROC_CH_DIR, exist_ok=True)

# ============================================================
# LOAD DATA FILES
# ============================================================
def load_data_dir(data_dir):
    snr_files = {}
    for f in os.listdir(data_dir):
        if f.endswith(".pth") and "snr" in f.lower():
            m = re.search(r"snr[_\-]?\s*(-?\d+)", f.lower())
            if m:
                snr_files[int(m.group(1))] = os.path.join(data_dir, f)
    return snr_files

snr_files = load_data_dir(DATA_DIR)
sorted_snrs = sorted(snr_files.keys())
train_snr = max([s for s in sorted_snrs if s <= 0])
TEST_SNR = min(sorted_snrs, key=lambda x: abs(x - DESIRED_TEST_SNR))

# ============================================================
# DATASET
# ============================================================
def get_2d_psd_data(data_dict):
    samples = data_dict["training data list"]
    labels = torch.stack(data_dict["training label list"])
    psd = [torch.mean(torch.stack(s), dim=0) for s in samples]
    return torch.stack(psd), labels.numpy()

class PSD2DDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    def __len__(self):
        return len(self.X) - SEQ_LEN
    def __getitem__(self, idx):
        return self.X[idx:idx+SEQ_LEN], torch.tensor(self.Y[idx+SEQ_LEN], dtype=torch.float32)

# ============================================================
# MODELS
# ============================================================
class DecoupleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 40, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(40, 160, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(160, NUM_CHANNELS)
    def forward(self, x):
        B,T,C,H,W = x.shape
        x = self.net(x.view(B*T,C,H,W)).squeeze()
        return self.fc(x).view(B,T,-1)[:, -1]

class CNN_GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1,64,3,padding=1)
        self.gru = nn.GRU(64,128,batch_first=True)
        self.fc = nn.Linear(128, NUM_CHANNELS)
    def forward(self, x):
        B,T,C,H,W = x.shape
        x = torch.relu(self.conv(x.view(B*T,C,H,W))).mean(dim=[2,3])
        g,_ = self.gru(x.view(B,T,-1))
        return self.fc(g[:,-1])

class HAPPO_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(64*20,256),
            nn.ReLU(),
            nn.Linear(256,NUM_CHANNELS)
        )
    def forward(self, x):
        return self.policy(x[:,-1].reshape(x.size(0),-1))

# ============================================================
# METRICS
# ============================================================
def compute_pd_pfa(y_true, y_score):
    Pd, Pfa = [], []
    for th in THRESHOLDS:
        y_pred = (y_score >= th).astype(int)
        TP = np.sum((y_pred==1)&(y_true==1))
        FN = np.sum((y_pred==0)&(y_true==1))
        FP = np.sum((y_pred==1)&(y_true==0))
        TN = np.sum((y_pred==0)&(y_true==0))
        Pd.append(TP/(TP+FN+1e-12))
        Pfa.append(FP/(FP+TN+1e-12))
    return np.array(Pd), np.array(Pfa)

def compute_auc(Pfa, Pd):
    idx = np.argsort(Pfa)
    return np.trapz(Pd[idx], Pfa[idx])

# ============================================================
# TRAIN & EVALUATE
# ============================================================
def train_and_evaluate():
    train_data = torch.load(snr_files[train_snr])
    X_train, Y_train = get_2d_psd_data(train_data)
    train_loader = DataLoader(PSD2DDataset(X_train,Y_train), BATCH_SIZE, shuffle=True)

    models = {
        "Decoupled CNN": DecoupleCNN().to(DEVICE),
        "CNN+GRU": CNN_GRU().to(DEVICE),
        "HAPPO": HAPPO_Model().to(DEVICE)
    }

    loss_fn = nn.BCEWithLogitsLoss()

    # -------- TRAIN --------
    for name, model in models.items():
        opt = optim.Adam(model.parameters(),1e-3)
        model.train()
        for _ in range(EPOCHS):
            for xb,yb in train_loader:
                xb,yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
        print(f"{name} trained")

    # -------- AUC vs SNR --------
    auc_rows = []
    for snr in sorted_snrs:
        data = torch.load(snr_files[snr])
        X,Y = get_2d_psd_data(data)
        loader = DataLoader(PSD2DDataset(X,Y), BATCH_SIZE)

        for name, model in models.items():
            scores=[]
            with torch.no_grad():
                for xb,_ in loader:
                    scores.append(torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy())

            scores = np.concatenate(scores).flatten()
            truth = Y[SEQ_LEN:].flatten()
            Pd,Pfa = compute_pd_pfa(truth, scores)
            auc_rows.append({"Model":name,"SNR":snr,"AUC":compute_auc(Pfa,Pd)})

    auc_df = pd.DataFrame(auc_rows)
    auc_df.to_csv(os.path.join(AUC_DIR,"AUC_vs_SNR.csv"),index=False)

    # ---- SINGLE AUC CURVE (ALL MODELS) ----
    plt.figure(figsize=(9,6))
    for name in auc_df.Model.unique():
        sub = auc_df[auc_df.Model==name]
        plt.plot(sub.SNR, sub.AUC, marker="o", linewidth=3, label=name)
    plt.xlabel("SNR (dB)")
    plt.ylabel("AUC")
    plt.title("AUC vs SNR (All Models)")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(AUC_DIR,"AUC_vs_SNR_All_Models.png"),dpi=300)
    plt.close()

    # -------- ROC PER CHANNEL (ALL MODELS TOGETHER) --------
    data = torch.load(snr_files[TEST_SNR])
    X,Y = get_2d_psd_data(data)
    loader = DataLoader(PSD2DDataset(X,Y), BATCH_SIZE)

    all_scores = {}
    for name, model in models.items():
        scores=[]
        with torch.no_grad():
            for xb,_ in loader:
                scores.append(torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy())
        all_scores[name] = np.concatenate(scores)

    truth = Y[SEQ_LEN:]

    for ch in range(NUM_CHANNELS):
        plt.figure(figsize=(7,6))
        for name in models.keys():
            Pd,Pfa = compute_pd_pfa(truth[:,ch], all_scores[name][:,ch])
            plt.plot(Pfa, Pd, linewidth=2.5, label=name)

        plt.xlabel("Pfa")
        plt.ylabel("Pd")
        plt.title(f"ROC – Channel {ch} (SNR={TEST_SNR} dB)")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(ROC_CH_DIR,f"ROC_Channel_{ch}_All_Models.png"),dpi=300)
        plt.close()

    print("✅ ALL RESULTS GENERATED")
    print("📂", RESULTS_DIR)

# ============================================================
if __name__ == "__main__":
    train_and_evaluate()
