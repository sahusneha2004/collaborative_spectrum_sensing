import os, re, torch, numpy as np, datetime, csv
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

NUM_CHANNELS = 128
SEQ_LEN = 8
BATCH_SIZE = 48
EPOCHS = 15
LR = 1e-4

# ================= PATH =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(SCRIPT_DIR, "GeneratedDatasets_realistic", "260103_001533")

# Output folder with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, f"results_{timestamp}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= LOAD DATA =================
def load_snr_files(path):
    snr_files = {}
    for f in os.listdir(path):
        if f.endswith(".pth") and "snr" in f.lower():
            m = re.search(r"snr[_\-]?\s*(-?\d+)", f.lower())
            if m:
                snr = float(m.group(1))
                snr_files[snr] = os.path.join(path, f)
    return snr_files

# ================= PSD EXTRACTION =================
def extract_psd(data_dict):
    samples = data_dict["training data list"]
    labels = torch.stack(data_dict["training label list"])

    X, Y = [], []
    for i, sample in enumerate(samples):
        avg = torch.mean(torch.stack(sample), dim=0).flatten()
        avg = (avg - avg.mean()) / (avg.std() + 1e-6)

        avg = avg[:NUM_CHANNELS] if len(avg) > NUM_CHANNELS else \
              torch.nn.functional.pad(avg, (0, NUM_CHANNELS - len(avg)))

        label = labels[i].flatten()
        label = label[:NUM_CHANNELS] if len(label) > NUM_CHANNELS else \
                torch.nn.functional.pad(label, (0, NUM_CHANNELS - len(label)))

        X.append(avg)
        Y.append(label)

    return torch.stack(X), torch.stack(Y)

# ================= DATASET =================
class PSDDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X) - SEQ_LEN

    def __getitem__(self, i):
        x_seq = self.X[i:i+SEQ_LEN]

        y_curr = self.Y[i+SEQ_LEN-1]  # detection
        y_fut  = self.Y[i+SEQ_LEN]    # prediction

        return x_seq, y_curr, y_fut

# ================= MODEL =================
class CNN_LSTM_Model(nn.Module):
    def __init__(self):
        super().__init__()

        # Conv2D but safe (no width collapse)
        self.conv1 = nn.Conv2d(1, 32, (3,1), padding=(1,0))
        self.conv2 = nn.Conv2d(32, 64, (3,1), padding=(1,0))
        self.conv3 = nn.Conv2d(64, 128, (3,1), padding=(1,0))

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d((2,1))  # only reduce channel dimension
        self.adapt = nn.AdaptiveAvgPool2d((2,1))

        self.feature_dim = 128 * 2 * 1

        self.lstm = nn.LSTM(self.feature_dim, 128, batch_first=True)

        self.det_head = nn.Linear(128, NUM_CHANNELS)
        self.pred_head = nn.Linear(128, NUM_CHANNELS)

    def forward(self, x):
        B, T, C = x.shape

        x = x.unsqueeze(2).unsqueeze(3)  # (B,T,1,C,1)
        x = x.view(B*T, 1, C, 1)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))

        x = self.adapt(x)
        x = x.view(B, T, -1)

        lstm_out, _ = self.lstm(x)
        h_t = lstm_out[:, -1, :]

        det_out = self.det_head(h_t)
        pred_out = self.pred_head(h_t)

        return det_out, pred_out

# ================= TRAIN =================
def train(model, loader):
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for xb, y_curr, y_fut in loader:
            xb = xb.to(DEVICE)
            y_curr = y_curr.to(DEVICE)
            y_fut = y_fut.to(DEVICE)

            optimizer.zero_grad()

            out_curr, out_fut = model(xb)

            loss = criterion(out_curr, y_curr) + \
                   criterion(out_fut, y_fut)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

# ================= CSV OUTPUT =================
def generate_csv(model, loader, filename):
    model.eval()

    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["det_mean", "pred_mean"])

        with torch.no_grad():
            for xb, _, _ in loader:
                xb = xb.to(DEVICE)

                out_curr, out_fut = model(xb)

                det = torch.sigmoid(out_curr).cpu().numpy()
                pred = torch.sigmoid(out_fut).cpu().numpy()

                for i in range(len(det)):
                    writer.writerow([
                        det[i].mean(),
                        pred[i].mean()
                    ])

    print(f"Saved: {filepath}")

# ================= MAIN =================
print("Loading data...")

snr_files = load_snr_files(BASE_PATH)

datasets = []
for snr in snr_files:
    data = torch.load(snr_files[snr], map_location="cpu", weights_only=True)
    X, Y = extract_psd(data)
    datasets.append(PSDDataset(X, Y))

full_dataset = ConcatDataset(datasets)
train_loader = DataLoader(full_dataset, BATCH_SIZE, shuffle=True)

model = CNN_LSTM_Model().to(DEVICE)

print("Training...")
train(model, train_loader)

print("Generating outputs...")
generate_csv(model, train_loader, "output.csv")

print(f"Done. Results stored in: {OUTPUT_DIR}")