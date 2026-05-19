import time
import torch
import numpy as np

############################################
# 1. LOAD MODEL (SAME AS TRAINED)
############################################

class PSDPresenceCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, padding=1)

        self.pool = torch.nn.MaxPool2d((2, 1))
        self.gap  = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc   = torch.nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.gap(x)
        return self.fc(x.flatten(1))


############################################
# 2. DEVICE
############################################

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

############################################
# 3. LOAD TRAINED MODEL WEIGHTS
############################################

model = PSDPresenceCNN().to(device)

# 👉 Load your trained weights here
# model.load_state_dict(torch.load("best_model.pth", map_location=device))

model.eval()

############################################
# 4. CREATE ONE PSD SNAPSHOT (SU INPUT)
############################################

# One PSD snapshot at SU
# Shape: [batch=1, channels=1, freq_bins=64, spectrum_channels=20]
psd_snapshot = torch.randn(1, 1, 64, 20).to(device)

############################################
# 5. WARM-UP (IMPORTANT FOR GPU)
############################################

with torch.no_grad():
    for _ in range(50):
        _ = model(psd_snapshot)

if device == "cuda":
    torch.cuda.synchronize()

############################################
# 6. MEASURE INFERENCE LATENCY
############################################

num_runs = 1000
latencies = []

with torch.no_grad():
    for _ in range(num_runs):
        start = time.perf_counter()

        _ = model(psd_snapshot)

        if device == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms

############################################
# 7. RESULTS
############################################

latencies = np.array(latencies)

print("\n================ INFERENCE LATENCY =================")
print(f"Mean latency   : {latencies.mean():.4f} ms")
print(f"Median latency : {np.median(latencies):.4f} ms")
print(f"Std deviation  : {latencies.std():.4f} ms")
print(f"Min latency    : {latencies.min():.4f} ms")
print(f"Max latency    : {latencies.max():.4f} ms")
print("===================================================")
