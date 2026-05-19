import os
import re
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, roc_auc_score, recall_score

############################################
# 0. CONFIG
############################################

DATASET_DIR = "/home/anjani/partial/sdr/GeneratedDatasets_realistic"
OUTPUT_DIR  = "/home/anjani/partial/sdr/Output"

NUM_CHANNELS = 20
EPOCHS = 8
BATCH_SIZE = 64
LR = 1e-3
TARGET_PFA = 0.1

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

############################################
# 1. HELPERS
############################################

def extract_snr(fname):
    m = re.search(r"SNR(-?\d+)", fname)
    return int(m.group(1)) if m else None

def find_threshold_for_pfa(y_true, y_prob, target_pfa=0.1):
    for th in np.linspace(0.05, 0.99, 300):
        y_pred = (y_prob >= th).astype(int)
        pfa = np.sum((y_pred == 1) & (y_true == 0)) / max(np.sum(y_true == 0), 1)
        if pfa <= target_pfa:
            return th
    return 0.9

############################################
# 2. DATASET
############################################

class EmulatorPSDDataset(Dataset):
    def __init__(self, root_dir, split="train", band_drop=6):
        self.X, self.y, self.snr = [], [], []

        files = []
        for r, _, f in os.walk(root_dir):
            for file in f:
                if file.endswith(".pth"):
                    files.append(os.path.join(r, file))

        print(f"[INFO] {split}: {len(files)} files")

        for file in files:
            snr_val = extract_snr(os.path.basename(file))
            data = torch.load(file, map_location="cpu")

            samples = data["training data list"] if split == "train" else data["testing data list"]
            labels  = data["training label list"] if split == "train" else data["testing label list"]

            for s, y in zip(samples, labels):
                psd = torch.stack(s).mean(0)
                psd = psd / (psd.max() + 1e-8)

                if split == "train":
                    drop = np.random.choice(NUM_CHANNELS, band_drop, replace=False)
                    psd[:, :, drop] = 0.0

                self.X.append(psd)
                self.y.append((torch.tensor(y) > 0).float())
                self.snr.append(snr_val)

        self.X = torch.stack(self.X)
        self.y = torch.stack(self.y)
        self.snr = np.array(self.snr)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.snr[idx]

############################################
# 3. MODEL
############################################

class PSDPresenceCNN(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d((2,1))
        self.out  = nn.Conv2d(128, num_channels, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.out(x)
        return x.mean(dim=[2,3])

############################################
# 4. TRAINING
############################################

train_ds = EmulatorPSDDataset(DATASET_DIR, "train")
test_ds  = EmulatorPSDDataset(DATASET_DIR, "test", band_drop=0)

train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds, BATCH_SIZE)

model = PSDPresenceCNN(NUM_CHANNELS).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.ones(NUM_CHANNELS).to(device)*1.5)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    loss_sum = 0
    for x,y,_ in train_loader:
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss={loss_sum/len(train_loader):.4f}")

############################################
# 5. INFERENCE
############################################

model.eval()
Y_true, Y_prob, SNRs = [], [], []

with torch.no_grad():
    for x,y,s in test_loader:
        prob = torch.sigmoid(model(x.to(device))).cpu().numpy()
        Y_true.append(y.numpy())
        Y_prob.append(prob)
        SNRs.append(s)

Y_true = np.vstack(Y_true)
Y_prob = np.vstack(Y_prob)
SNRs   = np.hstack(SNRs)

############################################
# 6. THRESHOLD CALIBRATION
############################################

thresholds = np.array([
    find_threshold_for_pfa(Y_true[:,c], Y_prob[:,c], TARGET_PFA)
    for c in range(NUM_CHANNELS)
])

Y_pred = (Y_prob >= thresholds).astype(int)

############################################
# 7. CHANNEL METRICS
############################################

Pd_ch = [recall_score(Y_true[:,c], Y_pred[:,c]) for c in range(NUM_CHANNELS)]
Pfa_ch = [
    np.sum((Y_pred[:,c]==1)&(Y_true[:,c]==0)) / max(np.sum(Y_true[:,c]==0),1)
    for c in range(NUM_CHANNELS)
]

############################################
# 8. CHANNEL HEATMAPS
############################################

plt.figure(figsize=(10,3))
plt.imshow(np.array(Pd_ch)[None,:], aspect="auto", cmap="viridis")
plt.colorbar(label="Pd")
plt.title("Channel-wise Pd (Pfa ≤ 0.1)")
plt.xlabel("Channel Index")
plt.yticks([])
plt.savefig(os.path.join(OUTPUT_DIR,"channel_pd_heatmap_fixed.pdf"))
plt.show()

plt.figure(figsize=(10,3))
plt.imshow(np.array(Pfa_ch)[None,:], aspect="auto", cmap="magma")
plt.colorbar(label="Pfa")
plt.title("Channel-wise Pfa")
plt.xlabel("Channel Index")
plt.yticks([])
plt.savefig(os.path.join(OUTPUT_DIR,"channel_pfa_heatmap_fixed.pdf"))
plt.show()

############################################
# 9. Pd vs Pfa (OPERATING CURVE)
############################################

pd_list, pfa_list = [], []
for th in np.linspace(0.05, 0.95, 40):
    Yp = (Y_prob >= th).astype(int)
    pd_list.append(np.mean([
        recall_score(Y_true[:,c], Yp[:,c], zero_division=0)
        for c in range(NUM_CHANNELS)
    ]))
    pfa_list.append(np.mean([
        np.sum((Yp[:,c]==1)&(Y_true[:,c]==0))/max(np.sum(Y_true[:,c]==0),1)
        for c in range(NUM_CHANNELS)
    ]))

plt.figure()
plt.plot(pfa_list, pd_list, marker="o")
plt.xlabel("Probability of False Alarm (Pfa)")
plt.ylabel("Probability of Detection (Pd)")
plt.title("Pd vs Pfa (Operating Characteristic)")
plt.grid()
plt.savefig(os.path.join(OUTPUT_DIR,"pd_vs_pfa.pdf"))
plt.show()

############################################
# 10. GLOBAL ROC
############################################

y_true_global = (Y_true.sum(axis=1) > 0).astype(int)
y_prob_global = 1 - np.prod(1 - Y_prob, axis=1)

fpr, tpr, _ = roc_curve(y_true_global, y_prob_global)
auc = roc_auc_score(y_true_global, y_prob_global)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
plt.plot([0,1],[0,1],'')
plt.xlabel("Pfa")
plt.ylabel("Pd")
plt.title("Global ROC (PU Presence)")
plt.legend()
plt.grid()
plt.savefig(os.path.join(OUTPUT_DIR,"roc_global_fixed.pdf"))
plt.show()

############################################
# 11. SNR vs Pd
############################################

snr_levels = sorted(set(SNRs[SNRs!=None]))
Pd_snr = []

for snr in snr_levels:
    idx = np.where(SNRs == snr)[0]
    Pd_snr.append(np.mean(
        (y_prob_global[idx] >= 0.5) &
        (y_true_global[idx] == 1)
    ))

plt.figure()
plt.plot(snr_levels, Pd_snr, marker="o")
plt.xlabel("SNR (dB)")
plt.ylabel("Probability of Detection")
plt.title("SNR vs Pd")
plt.grid()
plt.savefig(os.path.join(OUTPUT_DIR,"snr_vs_pd_fixed.pdf"))
plt.show()

print("\n✅ All graphs saved in:", OUTPUT_DIR)
