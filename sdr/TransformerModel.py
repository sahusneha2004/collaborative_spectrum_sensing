import os, re, torch, numpy as np, datetime
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
import matplotlib.pyplot as plt

# ================= CONFIG =================
BASE_PATH="/home/anjani/partial/sdr/GeneratedDatasets_realistic/260103_001533"
DEVICE="cuda" if torch.cuda.is_available() else "cpu"

NUM_CHANNELS=128
SEQ_LEN=10
BATCH_SIZE=32
EPOCHS=20
LR=1e-4

TIME_TAG=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
RESULT_DIR=os.path.join(BASE_PATH,f"SNR_RESULTS_{TIME_TAG}")
os.makedirs(RESULT_DIR,exist_ok=True)

# ================= LOAD =================
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

# ================= SNR STRATEGIES =================
SNR_STRATEGIES={
    "LOW_SNR":{"train":[-16,-14,-12],"test":[0,2,4,8,10]},
    "MIXED_SNR":{"train":[-16,-10,-5,0],"test":[-2,2,6,10]},
    "HIGH_SNR":{"train":[0,2,4],"test":[6,8,10]}
}

# ================= PREPROCESS =================
def normalize_psd(x):
    x=x+1e-8
    x=x/torch.max(x)
    return 10*torch.log10(x)

def apply_partial_observation(x,ratio=0.6):
    SU,_,F,C=x.shape
    mask=torch.zeros_like(x)
    for su in range(SU):
        visible=np.random.choice(C,int(C*ratio),replace=False)
        mask[su,:,:,visible]=1
    return x*mask

def extract_psd(data_dict):
    samples=data_dict["training data list"]
    labels=torch.stack(data_dict["training label list"])

    X=[];Y=[]
    for i,sample in enumerate(samples):
        su_tensor=torch.stack(sample)
        su_tensor=normalize_psd(su_tensor)
        su_tensor=apply_partial_observation(su_tensor)

        if su_tensor.shape[-1]<NUM_CHANNELS:
            su_tensor=torch.nn.functional.pad(su_tensor,(0,NUM_CHANNELS-su_tensor.shape[-1]))

        X.append(su_tensor)

        label=labels[i]
        if label.shape[-1]<NUM_CHANNELS:
            label=torch.nn.functional.pad(label,(0,NUM_CHANNELS-label.shape[-1]))

        Y.append(label)

    return torch.stack(X),torch.stack(Y)

# ================= DATASET =================
class PSDDataset(Dataset):
    def __init__(self,X,Y):
        self.X=X; self.Y=Y
    def __len__(self):
        return len(self.X)-SEQ_LEN
    def __getitem__(self,i):
        return self.X[i:i+SEQ_LEN], self.Y[i+SEQ_LEN]

# ================= BUILD LOADERS =================
def build_dataloaders(train_snrs,test_snrs):

    train_sets=[]; test_sets=[]

    for snr in train_snrs:
        if snr not in snr_files: continue
        data=torch.load(snr_files[snr],map_location="cpu")
        X,Y=extract_psd(data)
        if len(X)>SEQ_LEN:
            train_sets.append(PSDDataset(X,Y))

    for snr in test_snrs:
        if snr not in snr_files: continue
        data=torch.load(snr_files[snr],map_location="cpu")
        X,Y=extract_psd(data)
        if len(X)>SEQ_LEN:
            test_sets.append(PSDDataset(X,Y))

    train_loader=DataLoader(ConcatDataset(train_sets),BATCH_SIZE,shuffle=True)
    test_loader=DataLoader(ConcatDataset(test_sets),BATCH_SIZE)

    return train_loader,test_loader

# ================= MODELS =================
class CNN_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Conv2d(1,64,3,padding=1)
        self.lstm=nn.LSTM(64,128,batch_first=True)
        self.fc=nn.Linear(128,NUM_CHANNELS)

    def forward(self,x):
        B,T,SU,C,F,CH=x.shape
        x=x.mean(dim=2)
        x=x.view(B*T,C,F,CH)
        x=torch.relu(self.conv(x)).mean([2,3])
        x=x.view(B,T,-1)
        out,_=self.lstm(x)
        return self.fc(out[:,-1])

class CNN_Decoupling(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared=nn.Sequential(nn.Linear(NUM_CHANNELS,256),nn.ReLU())
        self.heads=nn.ModuleList([nn.Linear(256,1) for _ in range(NUM_CHANNELS)])

    def forward(self,x):
        x=x.mean(dim=4).squeeze(2).mean(dim=2)
        x=self.shared(x[:,-1])
        return torch.cat([h(x) for h in self.heads],dim=1)

class SpectrumTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Linear(NUM_CHANNELS,256)
        enc=nn.TransformerEncoderLayer(256,8,512,batch_first=True)
        self.trans=nn.TransformerEncoder(enc,4)
        self.fc=nn.Linear(256,NUM_CHANNELS)

    def forward(self,x):
        x=x.mean(dim=4).squeeze(2).mean(dim=2)
        x=self.embed(x)
        x=self.trans(x)
        return self.fc(x[:,-1])

class ProposedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Linear(NUM_CHANNELS,256)
        enc=nn.TransformerEncoderLayer(256,8,1024,dropout=0.1,batch_first=True)
        self.trans=nn.TransformerEncoder(enc,6)
        self.fc=nn.Linear(256,NUM_CHANNELS)

    def forward(self,x):
        x=x.mean(dim=4).squeeze(2).mean(dim=2)
        x=self.embed(x)
        x=self.trans(x)
        return self.fc(x[:,-1])

# ================= TRAIN =================
def train_model(model,loader):
    optimizer=optim.Adam(model.parameters(),lr=LR)
    criterion=nn.BCEWithLogitsLoss()

    losses=[]

    for epoch in range(EPOCHS):
        model.train()
        total=0

        for xb,yb in loader:
            xb,yb=xb.to(DEVICE),yb.to(DEVICE)
            optimizer.zero_grad()
            loss=criterion(model(xb),yb)
            loss.backward()
            optimizer.step()
            total+=loss.item()

        losses.append(total/len(loader))
        print(f"Epoch {epoch+1} Loss {losses[-1]:.4f}")

    return losses

# ================= EVAL =================
def get_scores(model,loader):
    model.eval()
    scores=[]; truth=[]
    with torch.no_grad():
        for xb,yb in loader:
            out=torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy()
            scores.append(out); truth.append(yb.numpy())
    return np.concatenate(truth).flatten(),np.concatenate(scores).flatten()

# ================= SNR CURVE =================
def evaluate_per_snr(model,test_snrs):
    snr_auc={}
    for snr in test_snrs:
        if snr not in snr_files: continue
        data=torch.load(snr_files[snr],map_location="cpu")
        X,Y=extract_psd(data)
        if len(X)<=SEQ_LEN: continue
        loader=DataLoader(PSDDataset(X,Y),BATCH_SIZE)
        y_true,y_score=get_scores(model,loader)
        snr_auc[snr]=roc_auc_score(y_true,y_score)
    return snr_auc

# ================= RUN =================
def run_experiment(model,name,train_loader,test_loader,strategy):

    folder=os.path.join(RESULT_DIR,strategy,name)
    os.makedirs(folder,exist_ok=True)

    model=model.to(DEVICE)

    print(f"\n🚀 {name} | {strategy}")

    losses=train_model(model,train_loader)

    y_true,y_score=get_scores(model,test_loader)
    auc=roc_auc_score(y_true,y_score)
    acc=accuracy_score(y_true,(y_score>0.5))

    # LOSS CURVE
    plt.figure()
    plt.plot(losses)
    plt.title("Loss")
    plt.savefig(os.path.join(folder,"loss.png"))
    plt.close()

    # ROC
    fpr,tpr,_=roc_curve(y_true,y_score)
    plt.figure()
    plt.plot(fpr,tpr,label=f"AUC={auc:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.legend()
    plt.savefig(os.path.join(folder,"roc.png"))
    plt.close()

    # SNR CURVE
    snr_auc=evaluate_per_snr(model,SNR_STRATEGIES[strategy]["test"])
    plt.figure()
    plt.plot(list(snr_auc.keys()),list(snr_auc.values()),marker='o')
    plt.title("AUC vs SNR")
    plt.savefig(os.path.join(folder,"snr_curve.png"))
    plt.close()

    with open(os.path.join(folder,"results.txt"),"w") as f:
        f.write(f"AUC:{auc}\nACC:{acc}\nSNR:{snr_auc}")

# ================= EXECUTION =================
models={
    "CNN_LSTM":CNN_LSTM(),
    "CNN_DECOUPLING":CNN_Decoupling(),
    "TRANSFORMER":SpectrumTransformer(),
    "PROPOSED":ProposedModel()
}

for strategy_name,snr_cfg in SNR_STRATEGIES.items():

    print(f"\n===== {strategy_name} =====")

    train_loader,test_loader=build_dataloaders(
        snr_cfg["train"],
        snr_cfg["test"]
    )

    for name,model in models.items():
        run_experiment(model,name,train_loader,test_loader,strategy_name)

print("\n✅ ALL RESULTS SAVED WITH CURVES")