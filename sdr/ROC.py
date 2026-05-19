import os
import re
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc
)

############################################
# 1. HELPER
############################################

def extract_snr(fname):
    m = re.search(r"SNR(-?\d+)", fname)
    return int(m.group(1)) if m else None


############################################
# 2. DATASET
############################################

class EmulatorPSDDataset(Dataset):
    """
    PU presence detection dataset
    Keeps band-level labels ONLY for evaluation
    """

    def __init__(self, root_dir, split="train", band_drop=6):
        self.X = []
        self.y_presence = []
        self.y_band = []
        self.snr = []

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

                if split == "train":
                    drop = np.random.choice(20, band_drop, replace=False)
                    su_psd[:, :, drop] = 0.0

                self.X.append(su_psd)
                self.y_presence.append(float(y.sum() > 0))
                self.y_band.append(y)
                self.snr.append(snr_val)

        self.X = torch.stack(self.X)
        self.y_presence = torch.tensor(self.y_presence).float()
        self.y_band = torch.stack(self.y_band)
        self.snr = np.array(self.snr)

    def __len__(self):
        return len(self.y_presence)

    def __getitem__(self, idx):
        return self.X[idx], self.y_presence[idx], self.y_band[idx], self.snr[idx]


############################################
# 3. MODEL
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
# 4. TRAIN + EVALUATE + ROC
############################################

def main():
    DATASET_DIR = "/home/anjani/partial/sdr/GeneratedDatasets_realistic"
    OUTPUT_DIR = "Output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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

        for x, y_presence, _, _ in train_loader:
            x = x.to(device)
            y_presence = y_presence.to(device)

            optimizer.zero_grad()
            loss = criterion(model(x), y_presence)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

        print(f"Epoch [{epoch+1}/5] | Train Loss: {loss_sum/len(train_loader):.4f}")

    ########################################
    # Evaluation
    ########################################

    model.eval()
    y_true = []
    y_prob = []
    y_band_all = []

    with torch.no_grad():
        for x, y_presence, y_band, _ in test_loader:
            probs = torch.sigmoid(model(x.to(device)))
            y_true.extend(y_presence.numpy())
            y_prob.extend(probs.cpu().numpy())
            y_band_all.append(y_band.numpy())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_band_all = np.vstack(y_band_all)

    ########################################
    # Overall metrics
    ########################################

    threshold = 0.5
    y_pred = (y_prob >= threshold).astype(int)

    print("\n================ OVERALL METRICS =================")
    print("Precision :", precision_score(y_true, y_pred))
    print("Recall    :", recall_score(y_true, y_pred))
    print("F1-score  :", f1_score(y_true, y_pred))
    print("ROC-AUC   :", roc_auc_score(y_true, y_prob))
    print("=================================================")

    ########################################
    # ROC: P_FA vs P_D
    ########################################

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)

    plt.xlabel("Probability of False Alarm (P_FA)")
    plt.ylabel("Probability of Detection (P_D)")
    plt.title("ROC Curve for PU Presence Detection")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right")

    roc_path = os.path.join(OUTPUT_DIR, "ROC_PFA_vs_PD.png")
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[INFO] ROC curve saved at: {roc_path}")

    ########################################
    # CHANNEL-WISE METRIC MATRIX
    ########################################

    num_channels = y_band_all.shape[1]
    metric_matrix = np.zeros((num_channels, 3))

    for ch in range(num_channels):
        y_true_ch = y_band_all[:, ch]

        if y_true_ch.sum() == 0:
            metric_matrix[ch, :] = np.nan
        else:
            metric_matrix[ch, 0] = precision_score(y_true_ch, y_pred, zero_division=0)
            metric_matrix[ch, 1] = recall_score(y_true_ch, y_pred, zero_division=0)
            metric_matrix[ch, 2] = f1_score(y_true_ch, y_pred, zero_division=0)

    print("\nCHANNEL-WISE PERFORMANCE MATRIX")
    print("Rows : Channels | Cols : [Precision, Recall, F1]\n")

    for ch in range(num_channels):
        print(
            f"Channel-{ch+1:02d} : "
            f"{metric_matrix[ch, 0]:.2f}, "
            f"{metric_matrix[ch, 1]:.2f}, "
            f"{metric_matrix[ch, 2]:.2f}"
        )

    return metric_matrix


############################################
# 5. RUN
############################################

if __name__ == "__main__":
    metric_matrix = main()
