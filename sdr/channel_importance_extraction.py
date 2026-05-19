"""
spectrum_explainability.py
==========================
Explains WHAT FREQUENCY CHANNELS / FEATURES the CNN_Decoupling model
focuses on when making a spectrum-occupancy prediction.

Techniques used (adapted from explainable-cnn concepts):
  1. Vanilla Saliency Map       — gradient of output w.r.t. input PSD
  2. Grad-CAM (conv layer)      — gradient-weighted feature map activations
  3. Guided Backpropagation     — only positive gradients through ReLUs
  4. Channel Importance Bar     — mean absolute gradient per frequency bin

All methods work on your 1D PSD input (shape: B x T x NUM_CHANNELS),
NOT on images, so they are directly adapted to spectrum sensing.

Usage
-----
  python spectrum_explainability.py

Requirements
------------
  pip install torch numpy matplotlib
  (No explainable-cnn install needed — we re-implement the core ideas.)
"""

import os, re, torch, numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader

# ── copy your constants & model definition here ──────────────────────────────
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CHANNELS = 128
SEQ_LEN      = 8
BATCH_SIZE   = 48
BASE_PATH    = "/mnt/110e4fe8-b416-4f34-8f8b-66a64d07a44f/anjani/partial/sdr/GeneratedDatasets_realistic/260103_001533"
OUTPUT_DIR   = "/mnt/110e4fe8-b416-4f34-8f8b-66a64d07a44f/anjani/partial/sdr/20260330_121222"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── model (same as your training script) ─────────────────────────────────────
class CNN_Decoupling(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2))
        )
        self.fc = nn.Linear(128 * 2 * 2, NUM_CHANNELS)

    def forward(self, x):
        # x: (B, T, 1, H, W)  — but here H=1, W=NUM_CHANNELS (1-D PSD as 2-D)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.net(x)
        x = x.view(B * T, -1)
        x = self.fc(x)
        return x.view(B, T, -1)[:, -1]   # (B, NUM_CHANNELS)


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

# ── data helpers (same as your training script) ───────────────────────────────
def load_snr_files(path):
    snr_files = {}
    for f in os.listdir(path):
        if f.endswith(".pth") and "snr" in f.lower():
            m = re.search(r"snr[_\-]?\s*(-?\d+)", f.lower())
            if m:
                snr = float(m.group(1))
                snr_files[snr] = os.path.join(path, f)
    return snr_files


def extract_psd(data_dict):
    samples = data_dict["training data list"]
    labels  = torch.stack(data_dict["training label list"])
    X, Y = [], []
    for i, sample in enumerate(samples):
        avg = torch.mean(torch.stack(sample), dim=0)
        avg = (avg - avg.mean()) / (avg.std() + 1e-6)
        if avg.shape[-1] < NUM_CHANNELS:
            avg = nn.functional.pad(avg, (0, NUM_CHANNELS - avg.shape[-1]))
        X.append(avg)
        label = labels[i]
        if label.shape[-1] < NUM_CHANNELS:
            label = nn.functional.pad(label, (0, NUM_CHANNELS - label.shape[-1]))
        Y.append(label)
    return torch.stack(X), torch.stack(Y)

def to_1d(data):
    if data.ndim == 3:
        data = data.squeeze(0)  # remove batch → (64,128)
    if data.ndim == 2:
        data = data.mean(axis=0)  # collapse height → (128,)
    return data

class PSDDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X, self.Y = X, Y

    def __len__(self):
        return len(self.X) - SEQ_LEN

    def __getitem__(self, i):
        return self.X[i:i + SEQ_LEN], self.Y[i + SEQ_LEN]


# ════════════════════════════════════════════════════════════════════════════
#  EXPLAINABILITY METHODS
# ════════════════════════════════════════════════════════════════════════════

class SpectrumExplainer:
    """
    Wraps a trained CNN_Decoupling model and provides four
    explainability methods adapted from explainable-cnn concepts
    to work on 1-D PSD spectrum inputs.

    The input tensor fed to each method has shape:
        (1, T, 1, 1, NUM_CHANNELS)
    i.e. a single batch, T time-steps, 1 channel, height=1, width=NUM_CHANNELS.
    This lets CNN treat the PSD as a 1×128 "image".
    """

    def __init__(self, model: nn.Module):
        self.model = model.eval().to(DEVICE)

    # ── helper: reshape 1-D PSD sequence → CNN-compatible tensor ─────────────
    @staticmethod
    def _to_cnn_input(seq):
    # Case 1: (T, NUM_CHANNELS) → old assumption
        if seq.dim() == 2:
            seq = seq.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        # Case 2: (T, 1, H, W) → YOUR actual case
        elif seq.dim() == 4:
            seq = seq.unsqueeze(0)  # (1, T, 1, H, W)

        # Case 3: already correct
        elif seq.dim() == 5:
            pass

        else:
            raise ValueError(f"Unexpected input shape: {seq.shape}")

        return seq.float().to(DEVICE).requires_grad_(True)

    # ── 1. Vanilla Saliency ───────────────────────────────────────────────────
    def saliency_map(self, seq: torch.Tensor) -> np.ndarray:
        """
        Gradient of (sum of output) w.r.t. input PSD channels.
        High values → those frequency bins strongly drive the prediction.

        Returns: (NUM_CHANNELS,) numpy array
        """
        x = self._to_cnn_input(seq)
        out = self.model(x)          # (1, NUM_CHANNELS)
        target_idx = torch.argmax(out)
        out[0, target_idx].backward()

        # |gradient| averaged over time-steps, squeezed to 1-D
        grad = x.grad.data.abs()     # (1, T, 1, 1, NUM_CHANNELS)
        saliency = grad.squeeze().mean(dim=0).cpu().numpy()   # (NUM_CHANNELS,)
        if saliency.ndim > 1:        # guard for T=1 edge case
            saliency = saliency.mean(axis=0)
        return saliency / (saliency.max() + 1e-8)

    # ── 2. Grad-CAM on last conv layer ────────────────────────────────────────
    def grad_cam(self, seq: torch.Tensor) -> np.ndarray:
        """
        Registers hooks on the last Conv2d layer of self.model.net,
        computes gradient-weighted activation map, and projects it
        back to NUM_CHANNELS frequency bins.

        Returns: (NUM_CHANNELS,) numpy array
        """
        fmaps, grads = {}, {}

        # find last Conv2d in model.net
        last_conv = None
        last_name = None
        for name, m in self.model.net.named_modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
                last_name = name

        if last_conv is None:
            raise RuntimeError("No Conv2d found in model.net")

        # register hooks
        def fwd_hook(module, inp, out):
            fmaps["feat"] = out

        def bwd_hook(module, grad_in, grad_out):
            grads["grad"] = grad_out[0]

        h_fwd = last_conv.register_forward_hook(fwd_hook)
        h_bwd = last_conv.register_full_backward_hook(bwd_hook)

        x = self._to_cnn_input(seq)
        out = self.model(x)
        out.sum().backward()

        h_fwd.remove()
        h_bwd.remove()

        # weights = global-average-pooled gradients  (C_feat,)
        feat = fmaps["feat"]          # (B*T, C_feat, H', W')
        grad = grads["grad"]          # same shape
        weights = grad.mean(dim=(2, 3), keepdim=True)   # (B*T, C_feat, 1, 1)

        gcam = (feat * weights).sum(dim=1).clamp(min=0)  # (B*T, H', W')
        gcam = gcam.mean(dim=0)                           # (H', W')

        # interpolate to NUM_CHANNELS
        gcam = gcam.unsqueeze(0).unsqueeze(0)            # (1,1,H',W')
        gcam = nn.functional.interpolate(
            gcam, size=(1, NUM_CHANNELS), mode="bilinear", align_corners=False
        )
        gcam = gcam.squeeze().cpu().detach().numpy()      # (NUM_CHANNELS,)
        if gcam.ndim > 1:
            gcam = gcam.mean(axis=0)
        gcam = gcam - gcam.min()
        return gcam / (gcam.max() + 1e-8)

    # ── 3. Guided Backpropagation ─────────────────────────────────────────────
    def guided_backprop(self, seq: torch.Tensor) -> np.ndarray:
        """
        Only propagates positive gradients through ReLUs (guided BP).
        Highlights fine-grained frequency features that positively
        contribute to the prediction.

        Returns: (NUM_CHANNELS,) numpy array
        """
        handles = []

        def guided_hook(module, grad_in, grad_out):
            if isinstance(module, nn.ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)

        for m in self.model.modules():
            if isinstance(m, nn.ReLU):
                handles.append(m.register_full_backward_hook(guided_hook))

        x = self._to_cnn_input(seq)
        out = self.model(x)
        out.sum().backward()

        for h in handles:
            h.remove()

        grad = x.grad.data.clamp(min=0)   # (1, T, 1, 1, NUM_CHANNELS)
        gbp = grad.squeeze().mean(dim=0).cpu().numpy()
        if gbp.ndim > 1:
            gbp = gbp.mean(axis=0)
        return gbp / (gbp.max() + 1e-8)

    # ── 4. Channel Importance (mean |gradient| per frequency bin) ────────────
    def channel_importance(self, loader: DataLoader, n_batches: int = 10) -> np.ndarray:
        """
        Averages |saliency| across many samples to get stable
        per-frequency-bin importance scores.

        Returns: (NUM_CHANNELS,) numpy array
        """
        importance = np.zeros(NUM_CHANNELS)
        count = 0
        for i, (xb, _) in enumerate(loader):
            if i >= n_batches:
                break
            for j in range(min(4, len(xb))):          # explain 4 samples
                sal = self.saliency_map(xb[j])
                importance += sal
                count += 1
        return importance / (count + 1e-8)


# ════════════════════════════════════════════════════════════════════════════
#  VISUALISATION
# ════════════════════════════════════════════════════════════════════════════

def plot_all_explanations(saliency, gcam, gbp, importance,
                          snr_val, sample_input, output_dir):
    """
    Produces a 4-panel figure saved as a PDF:
      [0] Raw PSD (last time-step of the input sequence)
      [1] Vanilla Saliency
      [2] Grad-CAM
      [3] Guided Backprop
    Plus a separate bar chart for Channel Importance.
    """
    freq_bins = np.arange(NUM_CHANNELS)
    psd_last = sample_input[-1].squeeze().mean(axis=0).numpy()   # (NUM_CHANNELS,) last time-step

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Spectrum Explainability  |  SNR = {snr_val} dB", fontsize=14)
    gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    axes_info = [
        (gs[0, 0], psd_last,   "Input PSD (last time-step)",   "Normalised Power", "steelblue"),
        (gs[0, 1], saliency,   "Vanilla Saliency Map",          "Importance",       "darkorange"),
        (gs[1, 0], gcam,       "Grad-CAM (last conv layer)",    "Activation",       "crimson"),
        (gs[1, 1], gbp,        "Guided Backpropagation",        "Importance",       "seagreen"),
    ]

    for spec, data, title, ylabel, color in axes_info:
        ax = fig.add_subplot(spec)
        ax.plot(freq_bins, data, color=color, linewidth=1.5)
        ax.fill_between(freq_bins, data, alpha=0.25, color=color)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Frequency Channel Index")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, NUM_CHANNELS - 1)
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.savefig(os.path.join(output_dir, f"explanations_snr{int(snr_val)}.pdf"),
                bbox_inches="tight")
    plt.close()

    # ── Channel importance bar chart ──────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(14, 4))
    colors = plt.cm.RdYlGn(importance / (importance.max() + 1e-8))
    ax2.bar(freq_bins, importance, color=colors, width=1.0, edgecolor="none")
    ax2.set_title(f"Average Channel Importance across samples  |  SNR = {snr_val} dB",
                  fontsize=12)
    ax2.set_xlabel("Frequency Channel Index")
    ax2.set_ylabel("Mean |Saliency|")
    ax2.set_xlim(0, NUM_CHANNELS - 1)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.savefig(os.path.join(output_dir, f"channel_importance_snr{int(snr_val)}.pdf"),
                bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved plots for SNR {snr_val} dB")


def print_top_channels(importance: np.ndarray, top_k: int = 10):
    """
    Prints the top-k most important frequency channels to the terminal.
    """
    top_idx = np.argsort(importance)[::-1][:top_k]
    print("\n" + "=" * 50)
    print(f"  TOP-{top_k} FREQUENCY CHANNELS THE MODEL FOCUSES ON")
    print("=" * 50)
    print(f"  {'Rank':<6} {'Channel Index':<16} {'Importance Score'}")
    print("  " + "-" * 40)
    for rank, idx in enumerate(top_idx, 1):
        bar = "█" * int(importance[idx] / importance.max() * 20)
        print(f"  {rank:<6} {idx:<16} {importance[idx]:.4f}  {bar}")
    print("=" * 50 + "\n")


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def explain_cnn_decoupling():
    # ── load a trained model checkpoint if available, else use random weights ─
    model = CNN_Decoupling().to(DEVICE)
    ckpt_path = os.path.join(OUTPUT_DIR, "cnn_checkpoint.pth")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        print("No checkpoint found — using randomly initialised weights.")
        print("  → Train your model first and save with:")
        print("    torch.save(model.state_dict(), 'cnn_checkpoint.pth')\n")

    explainer = SpectrumExplainer(model)

    # ── load one SNR file to explain ─────────────────────────────────────────
    snr_files = load_snr_files(BASE_PATH)
    SNR_LIST  = sorted(snr_files.keys())

    if not SNR_LIST:
        print("ERROR: No SNR .pth files found under:", BASE_PATH)
        return

    # explain first available SNR (change to any desired SNR)
    for target_snr in SNR_LIST:   # pick the middle SNR
        print(f"\nGenerating explanations for SNR = {target_snr} dB ...")

        data    = torch.load(snr_files[target_snr], map_location="cpu", weights_only=True)
        X, Y    = extract_psd(data)
        dataset = PSDDataset(X, Y)
        loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        # pick one sample for per-sample explanations
        sample_seq, _ = dataset[0]   # (T, NUM_CHANNELS)

        print("  Computing Vanilla Saliency ...")
        saliency   = explainer.saliency_map(sample_seq)

        print("  Computing Grad-CAM ...")
        gcam       = explainer.grad_cam(sample_seq)

        print("  Computing Guided Backpropagation ...")
        gbp        = explainer.guided_backprop(sample_seq)

        print("  Computing Channel Importance (avg over dataset) ...")
        importance = explainer.channel_importance(loader, n_batches=20)

        # ── print top channels ────────────────────────────────────────────────────
        print_top_channels(importance, top_k=10)

        saliency = to_1d(saliency)
        gcam     = to_1d(gcam)
        gbp      = to_1d(gbp)

        # ── save plots ────────────────────────────────────────────────────────────
        plot_all_explanations(saliency, gcam, gbp, importance,
                            target_snr, sample_seq, OUTPUT_DIR)

        print(f"\nAll outputs saved to: {OUTPUT_DIR}")
        print("Files produced:")
        for f in os.listdir(OUTPUT_DIR):
            print(f"  {f}")

def explain_cnn_lstm():
    """
    Explain for all available SNRs using the CNN_LSTM model by loading the corresponding files,
    processing the data, and generating explanations.
    """
    snr_files = load_snr_files(BASE_PATH)
    model = CNN_LSTM().to(DEVICE)

    ckpt_path = os.path.join(OUTPUT_DIR, "cnn_lstm_checkpoint.pth")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        print("No checkpoint found — using randomly initialised weights.")
        print("  → Train your model first and save with:")
        print("    torch.save(model.state_dict(), 'cnn_lstm_checkpoint.pth')\n")

    for snr, file_path in snr_files.items():
        print(f"Processing SNR: {snr} with CNN_LSTM")
        data_dict = torch.load(file_path, map_location=DEVICE)
        X, Y = extract_psd(data_dict)
        dataset = PSDDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward pass through the model
            outputs = model(inputs)

            # Generate explanations (example: Vanilla Saliency)
            explainer = SpectrumExplainer(model)
            cnn_input = explainer._to_cnn_input(inputs[0])
            saliency = explainer.vanilla_saliency(cnn_input)

            # Save or visualize the explanation
            output_path = os.path.join(OUTPUT_DIR, f"CNN_LSTM_SNR_{snr}_batch_{batch_idx}_saliency.png")
            plt.figure(figsize=(10, 4))
            plt.plot(saliency.cpu().detach().numpy())
            plt.title(f"Vanilla Saliency for CNN_LSTM SNR {snr}, Batch {batch_idx}")
            plt.savefig(output_path)
            plt.close()

            print(f"Saved explanation for CNN_LSTM SNR {snr}, Batch {batch_idx} at {output_path}")

if __name__ == "__main__":

    explain_cnn_lstm()