import os, re, torch, numpy as np, datetime, csv
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import Counter
from torch.utils.data import Dataset, DataLoader, ConcatDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEQ_LEN = 8
BATCH_SIZE = 48
EPOCHS = 15
LR = 1e-4

BASE_PATH = "GeneratedDatasets_realistic/260103_001533"

# ================= TIMESTAMP =================
start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(".", start_time)
os.makedirs(output_dir, exist_ok=True)

# ================= LOAD =================
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

train_snrs = [0, 6, 12]
test_snrs = [s for s in SNR_LIST if s not in train_snrs]

# ================= PSD =================
def extract_psd(data_dict):
    samples = data_dict["training data list"]
    labels = torch.stack(data_dict["training label list"])

    X, Y = [], []
    for i, sample in enumerate(samples):
        avg = torch.mean(torch.stack(sample), dim=0)
        avg = (avg - avg.mean()) / (avg.std() + 1e-6)

        if avg.dim() == 1:
            avg = avg.unsqueeze(0)

        X.append(avg)
        Y.append(labels[i])

    return torch.stack(X), torch.stack(Y)

# ================= DATASET =================
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
    def __init__(self, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.ReLU(),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2,2))
        )
        self.fc = nn.Linear(128*2*2, out_dim)

    def forward(self,x):
        if x.dim() == 4:
            x = x.unsqueeze(2)

        B,T,C,H,W = x.shape
        x = x.view(B*T,C,H,W)

        x = self.net(x)
        x = x.view(B,T,-1)

        return self.fc(x[:,-1])

class CNN_LSTM(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2,2))
        )
        self.lstm = nn.LSTM(64*2*2,96,batch_first=True)
        self.fc = nn.Linear(96, out_dim)

    def forward(self,x):
        if x.dim() == 4:
            x = x.unsqueeze(2)

        B,T,C,H,W = x.shape
        x = x.view(B*T,C,H,W)

        x = self.conv(x)
        x = x.view(B,T,-1)

        g,_ = self.lstm(x)
        return self.fc(g[:,-1])

class TransformerModel(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2,2))
        )
        self.feature_dim = 64*2*2

        self.input_proj = nn.Linear(self.feature_dim,192)
        self.pos_emb = nn.Parameter(torch.randn(1,SEQ_LEN,192))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=192,nhead=6,dim_feedforward=384,
            dropout=0.2,batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.norm = nn.LayerNorm(192)
        self.fc = nn.Linear(192, out_dim)

    def forward(self,x):
        if x.dim() == 4:
            x = x.unsqueeze(2)

        B,T,C,H,W = x.shape
        x = x.view(B*T,C,H,W)

        x = self.conv(x)
        x = x.view(B,T,-1)

        x = self.input_proj(x) + self.pos_emb[:,:T,:]
        x = self.transformer(x)
        x = self.norm(x)

        return self.fc(x[:,-1])

# ================= TRAIN =================
def train(model, loader):
    opt = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):
        model.train()
        total=0

        for xb,yb in loader:
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)

            opt.zero_grad()
            loss = loss_fn(model(xb), yb)

            loss.backward()
            opt.step()
            total += loss.item()

        print(f"Epoch {epoch+1}: {total:.4f}")

# ================= FEATURE IMPORTANCE =================
def get_dominant_feature(model, sample):
    model.eval()

    xb = sample.unsqueeze(0).to(DEVICE)
    xb.requires_grad = True

    out = torch.sigmoid(model(xb))
    out.mean().backward()

    grad = xb.grad
    importance = (grad * xb).abs().detach().cpu().numpy().flatten()

    return int(np.argmax(importance))

# ================= CSV =================
def generate_csv(model, loader, filename):
    path = os.path.join(output_dir, filename)
    counter = Counter()

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        for xb,yb in loader:
            xb = xb.to(DEVICE)

            preds = torch.sigmoid(model(xb)).cpu().detach().numpy()

            for i in range(len(xb)):
                sample = xb[i]

                flat = sample.cpu().numpy().flatten()
                pred = preds[i].mean()

                dom = get_dominant_feature(model, sample)
                counter[dom]+=1

                writer.writerow([pred, dom] + flat.tolist())

    return counter

# ================= PLOT =================
def plot(counter,name):
    if not counter: return

    x=list(counter.keys())
    y=list(counter.values())

    plt.figure()
    plt.bar(x,y)
    plt.title(name)
    plt.savefig(os.path.join(output_dir,name+".png"))
    plt.close()

# ================= MAIN =================
datasets=[]
num_channels=None

for snr in train_snrs:
    data=torch.load(snr_files[snr],map_location="cpu",weights_only=True)
    X,Y=extract_psd(data)

    if num_channels is None:
        num_channels = Y.shape[1]

    datasets.append(PSDDataset(X,Y))

train_loader = DataLoader(ConcatDataset(datasets), BATCH_SIZE, shuffle=True)

models={
    "CNN":CNN(num_channels).to(DEVICE),
    "CNN_LSTM":CNN_LSTM(num_channels).to(DEVICE),
    "Transformer":TransformerModel(num_channels).to(DEVICE)
}

# TRAIN
for name,m in models.items():
    print("Training",name)
    train(m,train_loader)

# TEST + CSV
for snr in test_snrs:
    data=torch.load(snr_files[snr],map_location="cpu",weights_only=True)
    X,Y=extract_psd(data)
    loader=DataLoader(PSDDataset(X,Y),BATCH_SIZE)

    for name,m in models.items():
        counter=generate_csv(m,loader,f"{name}_snr_{snr}.csv")
        plot(counter,f"{name}_snr_{snr}")

print("Done. Outputs saved in:", output_dir)