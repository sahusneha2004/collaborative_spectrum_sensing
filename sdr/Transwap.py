import os, re, torch, numpy as np, datetime
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from sklearn.metrics import roc_auc_score, accuracy_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CHANNELS = 128
SEQ_LEN = 8        # reduced
BATCH_SIZE = 48    # larger batch
EPOCHS = 15        # reduced
LR = 1e-4

BASE_PATH = "/home/anjani/partial/sdr/GeneratedDatasets_realistic/260103_001533"

# ================= TIMESTAMP =================
start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join("/home/anjani/partial/sdr", start_time)
os.makedirs(output_dir, exist_ok=True)

# ================= DATA =================
def load_snr_files(path):
    snr_files = {}
    for f in os.listdir(path):
        if f.endswith(".pth") and "snr" in f.lower():
            m = re.search(r"snr[_\-]?\s*(-?\d+)", f.lower())
            if m:
                snr = float(m.group(1))
                snr_files[snr] = os.path.join(path, f)
    return snr_files

snr_files = load_snr_files(BASE_PATH)
SNR_LIST = sorted(snr_files.keys())
# Select specific SNRs for training
desired_train_snrs = [0, 6, 12]

train_snrs = []
for s in SNR_LIST:
    if any(abs(s - d) < 1e-3 for d in desired_train_snrs):
        train_snrs.append(s)

# Everything else is test
test_snrs = [s for s in SNR_LIST if s not in train_snrs]

print("Train SNRs:", train_snrs)
print("Test SNRs:", test_snrs)

# ================= PSD =================
def extract_psd(data_dict):
    samples = data_dict["training data list"]
    labels = torch.stack(data_dict["training label list"])

    X, Y = [], []
    for i, sample in enumerate(samples):
        avg = torch.mean(torch.stack(sample), dim=0)
        avg = (avg - avg.mean()) / (avg.std() + 1e-6)

        if avg.shape[-1] < NUM_CHANNELS:
            avg = torch.nn.functional.pad(avg, (0, NUM_CHANNELS - avg.shape[-1]))

        X.append(avg)

        label = labels[i]
        if label.shape[-1] < NUM_CHANNELS:
            label = torch.nn.functional.pad(label, (0, NUM_CHANNELS - label.shape[-1]))

        Y.append(label)

    return torch.stack(X), torch.stack(Y)

class PSDDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X) - SEQ_LEN

    def __getitem__(self, i):
        return self.X[i:i+SEQ_LEN], self.Y[i+SEQ_LEN]

# ================= MODELS =================

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((2,2))   # lighter
        )

        self.fc = nn.Linear(128*2*2, NUM_CHANNELS)

    def forward(self, x):
        B,T,C,H,W = x.shape
        x = x.view(B*T, C, H, W)

        x = self.net(x)
        x = x.view(B*T, -1)

        x = self.fc(x)
        return x.view(B, T, -1)[:, -1]

class CNN_LSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2,2))
        )

        self.lstm = nn.LSTM(64*2*2, 96, batch_first=True)  # reduced size
        self.fc = nn.Linear(96, NUM_CHANNELS)

    def forward(self, x):
        B,T,C,H,W = x.shape

        x = x.view(B*T, C, H, W)
        x = self.conv(x)

        x = x.view(B, T, -1)

        g,_ = self.lstm(x)
        return self.fc(g[:, -1])

class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2,2))
        )

        self.feature_dim = 64*2*2

        self.input_proj = nn.Linear(self.feature_dim, 192)

        self.pos_emb = nn.Parameter(torch.randn(1, SEQ_LEN, 192))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=192,
            nhead=6,
            dim_feedforward=384,
            dropout=0.2,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)  # reduced
        self.norm = nn.LayerNorm(192)
        self.fc = nn.Linear(192, NUM_CHANNELS)

    def forward(self, x):
        B,T,C,H,W = x.shape

        x = x.view(B*T, C, H, W)
        x = self.conv(x)

        x = x.view(B, T, -1)

        x = self.input_proj(x)
        x = x + self.pos_emb[:, :T, :]

        x = self.transformer(x)
        x = self.norm(x)

        return self.fc(x[:, -1])

# ================= TRAIN =================
def train(model, train_loader, val_loader):
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    trigger = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        if epoch % 2 == 0:
            model.eval()
            val_loss = 0

            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    loss = criterion(model(xb), yb)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            print(f"Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}")

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trigger = 0
            else:
                trigger += 1
                if trigger >= 4:
                    print("Early stopping")
                    break
        else:
            print(f"Epoch {epoch+1}: Train={train_loss:.4f}")

# ================= EVAL =================
def evaluate(model, loader):
    model.eval()
    scores, truth = [], []

    with torch.no_grad():
        for xb, yb in loader:
            out = torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy()
            scores.append(out)
            truth.append(yb.numpy())

    scores = np.concatenate(scores)
    truth = np.concatenate(truth)

    auc = roc_auc_score(truth.flatten(), scores.flatten())
    acc = accuracy_score(truth.flatten(), (scores.flatten() > 0.5))
    err = 1 - acc

    return auc, acc, err

# ================= MAIN =================
models = {
    "CNN": CNN().to(DEVICE),
    "CNN_LSTM": CNN_LSTM().to(DEVICE),
    "Transformer": TransformerModel().to(DEVICE)
}

results = {k: {"auc": [], "acc": [], "err": []} for k in models}

train_data = []
for snr in train_snrs:
    data = torch.load(snr_files[snr], map_location="cpu", weights_only=True)
    X, Y = extract_psd(data)
    train_data.append(PSDDataset(X, Y))

full_dataset = ConcatDataset(train_data)

val_size = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, BATCH_SIZE, num_workers=4, pin_memory=True)

for name, model in models.items():
    print("Training:", name)
    train(model, train_loader, val_loader)

for snr in test_snrs:
    data = torch.load(snr_files[snr], map_location="cpu", weights_only=True)
    X, Y = extract_psd(data)
    loader = DataLoader(PSDDataset(X, Y), BATCH_SIZE, num_workers=4)

    for name, model in models.items():
        auc, acc, err = evaluate(model, loader)
        results[name]["auc"].append(auc)
        results[name]["acc"].append(acc)
        results[name]["err"].append(err)

for metric in ["auc", "acc", "err"]:
    plt.figure()
    for name in models:
        plt.plot(test_snrs, results[name][metric], label=name)

    plt.legend()
    plt.xlabel("SNR")
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} vs SNR")

    plt.savefig(os.path.join(output_dir, f"{metric}_vs_snr.pdf"))

print(f"Fast run results saved in: {output_dir}")