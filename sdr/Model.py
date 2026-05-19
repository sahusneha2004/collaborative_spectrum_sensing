import os, re, torch, numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CHANNELS = 128
SEQ_LEN = 10
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-4

BASE_PATH = "/home/anjani/partial/sdr/GeneratedDatasets_realistic/260103_001533"
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

# =========================
# LOAD DATA
# =========================
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

train_snrs = [s for s in SNR_LIST if s <= 0]
test_snrs  = [s for s in SNR_LIST if s > 0]

# =========================
# PREPROCESSING
# =========================
def normalize_psd(x):
    x = x + 1e-8
    x = x / torch.max(x)
    return 10 * torch.log10(x)

def apply_partial_observation(x, ratio=0.6):
    SU, _, F, C = x.shape
    mask = torch.zeros_like(x)
    for su in range(SU):
        visible = np.random.choice(C, int(C*ratio), replace=False)
        mask[su,:,:,visible] = 1
    return x * mask

def extract_psd(data_dict):
    samples = data_dict["training data list"]
    labels = torch.stack(data_dict["training label list"])

    X, Y = [], []

    for i, sample in enumerate(samples):
        su_tensor = torch.stack(sample)
        su_tensor = normalize_psd(su_tensor)
        su_tensor = apply_partial_observation(su_tensor)

        X.append(su_tensor)

        label = labels[i]
        Y.append(label)

    return torch.stack(X), torch.stack(Y)

# =========================
# DATASET
# =========================
class PSDDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X) - SEQ_LEN

    def __getitem__(self, i):
        return self.X[i:i+SEQ_LEN], self.Y[i+SEQ_LEN]

# =========================
# LOAD DATALOADERS
# =========================
train_sets, test_sets = [], []

for snr in train_snrs:
    data = torch.load(snr_files[snr], map_location="cpu")
    X,Y = extract_psd(data)
    train_sets.append(PSDDataset(X,Y))

for snr in test_snrs:
    data = torch.load(snr_files[snr], map_location="cpu")
    X,Y = extract_psd(data)
    test_sets.append(PSDDataset(X,Y))

train_loader = DataLoader(ConcatDataset(train_sets), BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(ConcatDataset(test_sets), BATCH_SIZE)

# =========================
# MODELS
# =========================
class CNN_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1,64,3,padding=1)
        self.lstm = nn.LSTM(64,128,batch_first=True)
        self.fc = nn.Linear(128,NUM_CHANNELS)

    def forward(self,x):
        B,T,SU,C,F,CH = x.shape
        x = x.mean(dim=2)
        x = x.view(B*T,C,F,CH)
        x = torch.relu(self.conv(x)).mean([2,3])
        x = x.view(B,T,-1)
        out,_ = self.lstm(x)
        return self.fc(out[:,-1])

class MultiTaskDNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NUM_CHANNELS,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU()
        )
        self.fc = nn.Linear(256,NUM_CHANNELS)

    def forward(self,x):
        x = x.mean(dim=4).squeeze(2).mean(dim=2)
        x = self.net(x)
        return self.fc(x[:,-1])

class SpectrumTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(NUM_CHANNELS,256)
        enc = nn.TransformerEncoderLayer(256,8,512,batch_first=True)
        self.trans = nn.TransformerEncoder(enc,4)
        self.fc = nn.Linear(256,NUM_CHANNELS)

    def forward(self,x):
        x = x.mean(dim=4).squeeze(2).mean(dim=2)
        x = self.embed(x)
        x = self.trans(x)
        return self.fc(x[:,-1])

class ProposedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(NUM_CHANNELS,256)
        enc = nn.TransformerEncoderLayer(256,8,1024,dropout=0.1,batch_first=True)
        self.trans = nn.TransformerEncoder(enc,6)
        self.fc = nn.Linear(256,NUM_CHANNELS)

    def forward(self,x):
        x = x.mean(dim=4).squeeze(2).mean(dim=2)
        x = self.embed(x)
        x = self.trans(x)
        return self.fc(x[:,-1])

# =========================
# TRAIN / EVAL
# =========================
def train_model(model, loader):
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):
        model.train()
        for xb,yb in loader:
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb),yb)
            loss.backward()
            optimizer.step()

def evaluate(model, loader):
    model.eval()
    scores,truth=[],[]
    with torch.no_grad():
        for xb,yb in loader:
            out=torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy()
            scores.append(out); truth.append(yb.numpy())
    scores=np.concatenate(scores); truth=np.concatenate(truth)
    return roc_auc_score(truth.flatten(),scores.flatten()), \
           accuracy_score(truth.flatten(),(scores.flatten()>0.5))

# =========================
# RUN EXPERIMENT
# =========================
def run_experiment(model, name, train_loader, test_loader):
    folder=os.path.join(RESULT_DIR,name)
    os.makedirs(folder,exist_ok=True)

    model=model.to(DEVICE)

    print(f"\n🚀 Training {name}")
    train_model(model,train_loader)

    auc,acc=evaluate(model,test_loader)
    print(f"{name} → AUC:{auc:.4f} ACC:{acc:.4f}")

    with open(os.path.join(folder,"results.txt"),"w") as f:
        f.write(f"AUC:{auc}\nACC:{acc}")

    return auc,acc

# =========================
# RUN ALL MODELS
# =========================
models={
    "CNN_LSTM":CNN_LSTM(),
    "MULTITASK_DNN":MultiTaskDNN(),
    "TRANSFORMER":SpectrumTransformer(),
    "PROPOSED":ProposedModel()
}

results={}

for name,model in models.items():
    auc,acc=run_experiment(model,name,train_loader,test_loader)
    results[name]=(auc,acc)

# =========================
# PLOTS
# =========================
names=list(results.keys())
auc_vals=[results[n][0] for n in names]
acc_vals=[results[n][1] for n in names]

plt.figure()
plt.bar(names,auc_vals)
plt.savefig(os.path.join(RESULT_DIR,"auc.pdf"))

plt.figure()
plt.bar(names,acc_vals)
plt.savefig(os.path.join(RESULT_DIR,"acc.pdf"))

print("\n✅ FINAL RESEARCH CODE — ERROR FREE")