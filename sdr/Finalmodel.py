import os
import re
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score
)

############################################
# 1. HELPER: SNR EXTRACTION
############################################

def extract_snr(fname):
    m = re.search(r"SNR(-?\d+)", fname)
    return int(m.group(1)) if m else None


############################################
# 2. DATASET: PU PRESENCE DETECTION
############################################

class EmulatorPSDDataset(Dataset):
    def __init__(self, root_dir, split="train", band_drop=6):
        self.X, self.y, self.snr = [], [], []

        files = []
        for r, _, f in os.walk(root_dir):
            for file in f:
                if file.endswith(".pth"):
                    files.append(os.path.join(r, file))

        if len(files) == 0:
            raise RuntimeError("No .pth files found")

        print(f"[INFO] Found {len(files)} .pth files for split={split}")

        for file in files:
            snr_val = extract_snr(os.path.basename(file))
            data = torch.load(file, map_location="cpu")

            samples = data["training data list"] if split == "train" else data["testing data list"]
            labels  = data["training label list"] if split == "train" else data["testing label list"]

            for s, y in zip(samples, labels):
                su_psd = torch.stack(s).mean(0)
                su_psd = su_psd / (su_psd.max() + 1e-8)

                # Partial observation (training only)
                if split == "train":
                    drop = np.random.choice(20, band_drop, replace=False)
                    su_psd[:, :, drop] = 0.0

                pu_present = float(y.sum() > 0)

                self.X.append(su_psd)
                self.y.append(pu_present)
                self.snr.append(snr_val)

        self.X = torch.stack(self.X)
        self.y = torch.tensor(self.y).float()
        self.snr = np.array(self.snr)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.snr[idx]


############################################
# 3. CNN MODEL (BINARY OUTPUT)
############################################

class PSDPresenceCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        self.pool = nn.MaxPool2d((2, 1))
        self.gap  = nn.AdaptiveAvgPool2d((1, 1))
        self.fc   = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.gap(x)
        return self.fc(x.flatten(1)).squeeze(1)


############################################
# 4. TRAIN + EVALUATE
############################################

def main():
    DATASET_DIR = "/home/anjani/partial/sdr/GeneratedDatasets_realistic"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = EmulatorPSDDataset(DATASET_DIR, "train")
    test_ds  = EmulatorPSDDataset(DATASET_DIR, "test", band_drop=0)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=64)

    model = PSDPresenceCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("\n[INFO] Training started\n")

    for epoch in range(5):
        model.train()
        loss_sum = 0.0

        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        print(f"Epoch [{epoch+1}/5] | Train Loss: {loss_sum/len(train_loader):.4f}")

    ########################################
    # Evaluation
    ########################################

    model.eval()
    y_true, y_prob, snrs = [], [], []

    with torch.no_grad():
        for x, y, s in test_loader:
            probs = torch.sigmoid(model(x.to(device)))
            y_true.extend(y.numpy())
            y_prob.extend(probs.cpu().numpy())
            snrs.extend(s)

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    snrs   = np.array(snrs)

    # 🔑 FIX: DEFINE y_pred PROPERLY
    threshold = 0.5
    y_pred = (y_prob >= threshold).astype(int)

    precision = precision_score(y_true, y_pred)
    recall    = recall_score(y_true, y_pred)
    f1        = f1_score(y_true, y_pred)
    roc_auc  = roc_auc_score(y_true, y_prob)

    print("\n================ METRICS =================")
    print(f"Threshold : {threshold}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print("=========================================")

    ########################################
    # Pd vs SNR
    ########################################

    snr_levels = sorted(set(snrs[snrs != None]))
    Pd = []

    for snr in snr_levels:
        idx = np.where(snrs == snr)[0]
        Pd.append(recall_score(y_true[idx], y_pred[idx]))

    plt.figure()
    plt.plot(snr_levels, Pd, marker="o")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Probability of Detection")
    plt.title("SNR vs Pd (PU Presence Detection)")
    plt.grid()
    plt.show()

    ########################################
    # ROC Curve
    ########################################

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("Probability of False Alarm")
    plt.ylabel("Probability of Detection")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig("/home/anjani/partial/sdr/Output/roc_curvepart1.pdf")


############################################
# 5. RUN
############################################

if __name__ == "__main__":
    main()
