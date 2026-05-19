import os, re, torch, numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score, accuracy_score

DEVICE="cuda" if torch.cuda.is_available() else "cpu"

NUM_CHANNELS=128
SEQ_LEN=8              # increased (important)
BATCH_SIZE=32
EPOCHS=30              # increased
LR=1e-4

BASE_PATH="/home/anjani/partial/sdr/GeneratedDatasets_realistic/260103_001533"

# =========================
# 🔬 1. LOAD SNR FILES
# =========================
def load_snr_files(path):
    snr_files={}
    for f in os.listdir(path):
        if f.endswith(".pth") and "snr" in f.lower():
            m=re.search(r"snr[_\-]?\s*(-?\d+)",f.lower())
            if m:
                snr=float(m.group(1))
                snr_files[snr]=os.path.join(path,f)
    return snr_files

snr_files=load_snr_files(BASE_PATH)
SNR_LIST=sorted(snr_files.keys())

train_snrs=[s for s in SNR_LIST if s<=0]
test_snrs=[s for s in SNR_LIST if s>0]

# =========================
# 🔬 2. PSD NORMALIZATION
# =========================
def normalize_psd(x):
    x = x + 1e-8
    x = x / torch.max(x)
    x = 10 * torch.log10(x)
    return x

# =========================
# 🔬 3. DATA EXTRACTION (FIXED)
# =========================
def extract_psd(data_dict):
    samples=data_dict["training data list"]
    labels=torch.stack(data_dict["training label list"])

    X=[]; Y=[]
    for i,sample in enumerate(samples):

        # KEEP SU INFORMATION (NO averaging)
        su_tensor=torch.stack(sample)   # [SU,1,freq,channels]

        su_tensor=normalize_psd(su_tensor)

        if su_tensor.shape[-1] < NUM_CHANNELS:
            su_tensor=torch.nn.functional.pad(
                su_tensor,(0,NUM_CHANNELS-su_tensor.shape[-1])
            )

        X.append(su_tensor)

        label=labels[i]
        if label.shape[-1] < NUM_CHANNELS:
            label=torch.nn.functional.pad(label,(0,NUM_CHANNELS-label.shape[-1]))

        Y.append(label)

    return torch.stack(X), torch.stack(Y)

# =========================
# 🔬 4. DATASET
# =========================
class PSDDataset(Dataset):
    def __init__(self,X,Y):
        self.X=X; self.Y=Y

    def __len__(self):
        return len(self.X)-SEQ_LEN

    def __getitem__(self,i):
        return (
            self.X[i:i+SEQ_LEN],   # [T,SU,1,F,C]
            self.Y[i+SEQ_LEN]
        )

# =========================
# 🔬 5. RESEARCH TRANSFORMER
# =========================
class ResearchTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed = nn.Linear(NUM_CHANNELS,256)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=6
        )

        self.fc = nn.Linear(256, NUM_CHANNELS)

    def forward(self,x):
        # x: [B,T,SU,1,F,C]

        x = x.mean(dim=4)   # remove freq → [B,T,SU,1,C]
        x = x.squeeze(3)    # [B,T,SU,C]

        # fuse SU dimension (spatial pooling)
        x = x.mean(dim=2)   # [B,T,C]

        x = self.embed(x)
        x = self.transformer(x)

        return self.fc(x[:,-1])

# =========================
# 🔬 6. TRAIN
# =========================
def train(model,loader):

    optimizer=optim.Adam(model.parameters(),lr=LR)

    # weighted loss (important)
    pos_weight=torch.ones(NUM_CHANNELS)*5
    criterion=nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))

    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,T_max=EPOCHS
    )

    for epoch in range(EPOCHS):

        model.train()
        total_loss=0

        for xb,yb in loader:

            xb,yb=xb.to(DEVICE),yb.to(DEVICE)

            optimizer.zero_grad()

            out=model(xb)
            loss=criterion(out,yb)

            loss.backward()

            # gradient clipping (very important)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)

            optimizer.step()

            total_loss+=loss.item()

        scheduler.step()

        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

# =========================
# 🔬 7. EVALUATION
# =========================
def evaluate(model,loader):

    model.eval()
    scores=[]; truth=[]

    with torch.no_grad():
        for xb,yb in loader:
            out=torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy()
            scores.append(out); truth.append(yb.numpy())

    scores=np.concatenate(scores)
    truth=np.concatenate(truth)

    auc=roc_auc_score(truth.flatten(),scores.flatten())
    acc=accuracy_score(truth.flatten(),(scores.flatten()>0.5))
    err=1-acc

    return auc,acc,err

# =========================
# 🔬 8. LOAD TRAIN DATA
# =========================
train_data=[]
for snr in train_snrs:
    data=torch.load(snr_files[snr],map_location="cpu")
    X,Y=extract_psd(data)
    train_data.append(PSDDataset(X,Y))

train_loader=DataLoader(
    torch.utils.data.ConcatDataset(train_data),
    BATCH_SIZE,
    shuffle=True
)

# =========================
# 🔬 9. TRAIN MODEL
# =========================
model=ResearchTransformer().to(DEVICE)

print("🚀 Training Research Transformer")
train(model,train_loader)

# =========================
# 🔬 10. TEST
# =========================
results={"auc":[],"acc":[],"err":[]}

for snr in test_snrs:

    data=torch.load(snr_files[snr],map_location="cpu")
    X,Y=extract_psd(data)

    loader=DataLoader(PSDDataset(X,Y),BATCH_SIZE)

    auc,acc,err=evaluate(model,loader)

    results["auc"].append(auc)
    results["acc"].append(acc)
    results["err"].append(err)

# =========================
# 🔬 11. PLOTS
# =========================
for metric in ["auc","acc","err"]:
    plt.figure()
    plt.plot(test_snrs,results[metric],marker='o')
    plt.xlabel("SNR")
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} vs SNR (Research Model)")
    plt.grid()
    plt.savefig(f"research_{metric}.png")

print("✅ Research Model Complete")