import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score

# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = "/mnt/110e4fe8-b416-4f34-8f8b-66a64d07a44f/anjani/partial/PartialObservation-main/RefinedNewData/SNRs/PUswitch/260409_22_30"
SAVE_ROOT = "/mnt/110e4fe8-b416-4f34-8f8b-66a64d07a44f/anjani/partial/PartialObservation-main/final/PartialObservation/Saved_Models/Transformer/PUswitch"

# =========================================================
# DEVICE
# =========================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Using device: {DEVICE}')

# =========================================================
# HYPERPARAMETERS
# =========================================================

BATCH_SIZE = 64
EPOCHS = 30

LR = 1e-4

SNR_LIST = [-8]

NUM_SU = 10
NUM_CHANNELS = 20

EMBED_DIM = 128

# =========================================================
# DATASET
# =========================================================

class TransformerDataset(Dataset):

    def __init__(self, data_list, label_list):

        self.data_list = data_list
        self.label_list = label_list

    def __len__(self):

        return len(self.label_list)

    def __getitem__(self, idx):

        sample = self.data_list[idx]

        # shape:
        # [NUM_SU, 1, 64, 20]

        x = torch.stack(sample, dim=0).float()

        # ==========================================
        # PER SAMPLE NORMALIZATION
        # ==========================================

        mean = x.mean()
        std = x.std()

        x = (x - mean) / (std + 1e-8)

        y = self.label_list[idx].float()

        return x, y


# =========================================================
# CNN FEATURE EXTRACTOR
# =========================================================

class SUEncoder(nn.Module):

    def __init__(self, embed_dim):

        super().__init__()

        self.cnn = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),
            nn.GELU(),

            nn.Conv2d(
                32,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),
            nn.GELU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.fc = nn.Sequential(

            nn.Linear(
                64 * 4 * 4,
                embed_dim
            ),

            nn.LayerNorm(embed_dim),

            nn.GELU()
        )

    def forward(self, x):

        # x shape:
        # [B, S, 1, 64, 20]

        B, S, C, H, W = x.shape

        x = x.view(B * S, C, H, W)

        x = self.cnn(x)

        x = x.view(B * S, -1)

        x = self.fc(x)

        x = x.view(B, S, -1)

        return x


# =========================================================
# TRANSFORMER MODEL
# =========================================================

class TransformerModel(nn.Module):

    def __init__(
        self,
        seq_len,
        embed_dim,
        num_channels
    ):

        super().__init__()

        self.encoder = SUEncoder(embed_dim)

        # CLS TOKEN

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, embed_dim)
        )

        # POSITION EMBEDDING

        self.pos_emb = nn.Parameter(
            torch.randn(
                1,
                seq_len + 1,
                embed_dim
            ) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=embed_dim * 2,
            dropout=0.2,
            activation='gelu',
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.classifier = nn.Sequential(

            nn.Linear(embed_dim, 128),

            nn.GELU(),

            nn.Dropout(0.2),

            nn.Linear(128, num_channels)
        )

    def forward(self, x):

        x = self.encoder(x)

        B = x.size(0)

        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = torch.cat(
            [cls_tokens, x],
            dim=1
        )

        x = x + self.pos_emb

        x = self.transformer(x)

        cls_output = x[:, 0]

        out = self.classifier(cls_output)

        return out


# =========================================================
# DATA LOADING
# =========================================================

def load_data(file_path):

    data = torch.load(file_path)

    train_data = data['training data list']
    train_labels = data['training label list']

    test_data = data['testing data list']
    test_labels = data['testing label list']

    return (
        train_data,
        train_labels,
        test_data,
        test_labels
    )


# =========================================================
# ROC COMPUTATION
# =========================================================

def compute_pd_pfa(y_true, y_score):

    y_true = y_true.ravel()
    y_score = y_score.ravel()

    pfa, pd, thresholds = roc_curve(
        y_true,
        y_score
    )

    auc = roc_auc_score(
        y_true,
        y_score
    )

    return pd, pfa, auc


# =========================================================
# EVALUATION
# =========================================================

def evaluate(model, loader):

    model.eval()

    y_scores = []
    y_trues = []

    with torch.no_grad():

        for xb, yb in loader:

            xb = xb.to(DEVICE)

            out = model(xb)

            probs = torch.sigmoid(out)

            y_scores.append(probs.cpu())

            y_trues.append(yb)

    y_scores = torch.cat(
        y_scores,
        dim=0
    ).numpy()

    y_trues = torch.cat(
        y_trues,
        dim=0
    ).numpy()

    return y_trues, y_scores


# =========================================================
# TRAINING
# =========================================================

def train_epoch(
    model,
    loader,
    optimizer,
    criterion
):

    model.train()

    total_loss = 0.0

    for xb, yb in loader:

        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()

        out = model(xb)

        loss = criterion(out, yb)

        loss.backward()

        # ==========================================
        # GRADIENT CLIPPING
        # ==========================================

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# =========================================================
# MAIN SNR TRAINING
# =========================================================

def run_snr(snr, data_path):

    print('\n' + '=' * 60)
    print(f'TRAINING FOR SNR = {snr} dB')
    print('=' * 60)

    (
        train_data,
        train_labels,
        test_data,
        test_labels
    ) = load_data(data_path)

    # ==========================================
    # DATASETS
    # ==========================================

    train_dataset = TransformerDataset(
        train_data,
        train_labels
    )

    test_dataset = TransformerDataset(
        test_data,
        test_labels
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # ==========================================
    # SAVE DIRECTORY
    # ==========================================

    timestamp = datetime.datetime.now().strftime('%y%m%d_%H_%M')

    save_dir = os.path.join(
        SAVE_ROOT,
        f'{snr}dB_{timestamp}'
    )

    os.makedirs(save_dir, exist_ok=True)

    # ==========================================
    # MODEL
    # ==========================================

    model = TransformerModel(
        seq_len=NUM_SU,
        embed_dim=EMBED_DIM,
        num_channels=NUM_CHANNELS
    )

    model.to(DEVICE)

    # ==========================================
    # POSITIVE WEIGHT
    # ==========================================

    train_labels_tensor = torch.stack(
        train_labels
    ).float()

    positives = train_labels_tensor.sum(dim=0)

    negatives = (
        len(train_labels_tensor)
        - positives
    )

    pos_weight = negatives / (positives + 1e-8)

    pos_weight = pos_weight.to(DEVICE)

    # ==========================================
    # LOSS + OPTIMIZER
    # ==========================================

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    # ==========================================
    # TRAINING TRACKERS
    # ==========================================

    losses = []
    aucs = []

    # ==========================================
    # TRAIN LOOP
    # ==========================================

    for epoch in range(1, EPOCHS + 1):

        loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion
        )

        scheduler.step()

        losses.append(loss)

        # ======================================
        # EVALUATION
        # ======================================

        y_true, y_score = evaluate(
            model,
            test_loader
        )

        auc = roc_auc_score(
            y_true.ravel(),
            y_score.ravel()
        )

        aucs.append(auc)

        # ======================================
        # DEBUG PREDICTION STATS
        # ======================================

        print(
            f'Prediction stats -> '
            f'MIN: {y_score.min():.4f} | '
            f'MAX: {y_score.max():.4f} | '
            f'MEAN: {y_score.mean():.4f}'
        )

        print(
            f'SNR {snr} | '
            f'Epoch {epoch}/{EPOCHS} | '
            f'Loss {loss:.4f} | '
            f'ROC-AUC {auc:.4f}'
        )

    # =====================================================
    # FINAL ROC
    # =====================================================

    y_true, y_score = evaluate(
        model,
        test_loader
    )

    pd_list, pfa_list, auc = compute_pd_pfa(
        y_true,
        y_score
    )

    print(f'\nFINAL ROC-AUC FOR SNR {snr}: {auc:.4f}')

    # =====================================================
    # SAVE TRAINING METRICS
    # =====================================================

    conv_df = pd.DataFrame({
        'Loss': losses,
        'ROC_AUC': aucs
    })

    conv_df.to_excel(
        os.path.join(
            save_dir,
            f'Convergence_SNR{snr}.xlsx'
        ),
        index=False
    )

    # =====================================================
    # SAVE ROC
    # =====================================================

    roc_df = pd.DataFrame({
        'PFA': pfa_list,
        'PD': pd_list
    })

    roc_df.to_excel(
        os.path.join(
            save_dir,
            f'ROC_SNR{snr}.xlsx'
        ),
        index=False
    )

    # =====================================================
    # SAVE MODEL
    # =====================================================

    torch.save(
        model.state_dict(),
        os.path.join(
            save_dir,
            f'Transformer_SNR{snr}.pth'
        )
    )

    # =====================================================
    # PLOT ROC
    # =====================================================

    plt.figure(figsize=(6, 5))

    plt.plot(
        pfa_list,
        pd_list,
        linewidth=2
    )

    plt.xlabel('Probability of False Alarm (PFA)')

    plt.ylabel('Probability of Detection (PD)')

    plt.title(
        f'ROC Curve | SNR = {snr} dB | AUC = {auc:.4f}'
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            save_dir,
            f'ROC_SNR{snr}.png'
        )
    )

    plt.close()

    print(f'Results saved to:\n{save_dir}')


# =========================================================
# MAIN
# =========================================================

if __name__ == '__main__':

    os.makedirs(
        SAVE_ROOT,
        exist_ok=True
    )

    for snr in SNR_LIST:

        filename = f'Data_SNR{snr}vol1.pth'

        file_path = os.path.join(
            DATA_DIR,
            filename
        )

        if not os.path.exists(file_path):

            print(
                f'WARNING: Missing file {file_path}'
            )

            continue

        run_snr(
            snr,
            file_path
        )

    print('\n✅ Improved Transformer Training Complete.')