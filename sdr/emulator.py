import os
import math
import random
import datetime
import shutil
from copy import deepcopy


import torch
import numpy as np

# ---------------------------
# 1) Mount Google Drive
# ---------------------------
# Removed google.colab mount — running on a server now.


# ---------------------------
# 2) Configuration (edit these if needed)
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
DRIVE_BASE = '/home/anjani/partial/sdr'
MOD_FILES = {
    '16QAM': os.path.join(DRIVE_BASE, 'psd_log_16qam.pth'),
    '8PSK' : os.path.join(DRIVE_BASE, 'psd_log_8psk.pth'),
    'QPSK' : os.path.join(DRIVE_BASE, 'psd_log_QPSK.pth'),
    'BPSK' : os.path.join(DRIVE_BASE, 'psd_log_BPSK.pth'),
}


# Output folder
TIME_TAG = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
OUT_DIR = os.path.join(DRIVE_BASE, 'GeneratedDatasets_realistic', TIME_TAG)
os.makedirs(OUT_DIR, exist_ok=True)
print('Output will be saved to:', OUT_DIR)

# Generation parameters (change to make larger/smaller datasets)
NCH = 20 # number of narrowband channels (bands)
NW = 64 # freq points per band (PSDs are 192 points -> 3 bands of 64)
NPU = 10
VOL_TRAIN = 20 # samples per class for training (increase for bigger dataset)
VOL_TEST = 5 # samples per class for testing
DIST_AMP = 10 # distance amplifier used in pathloss formula
ALPHA = 3.71
BETA = 10**3.154


# SNR policy: we'll sample per-PU SNRs from the distributions found in each modulation file
# but also allow overriding with desired list (global) if you want
GLOBAL_SNR_LIST = None # e.g., [-2, -4, -6] OR None to rely solely on recorded SNRs


# Frequency drift range (in fraction of band) e.g. +/- 0.1 bands (so shift up to 6.4 pts)
FREQ_DRIFT_FRAC = 0.08


# Shadowing std (dB)
SHADOW_STD_DB = 3.5


# Random seed
RND_SEED = 42
random.seed(RND_SEED)
np.random.seed(RND_SEED)

# ---------------------------
# 3) Helpers to load received .pth files and extract PSDs + SNRs
# ---------------------------

def load_received_file(path):
    """Load .pth file saved with torch.save and return dict with 'psds' (Nx192), 'snrs' (N) if available."""
    # Explicitly set weights_only=False to handle older .pth files that might contain numpy objects
    obj = torch.load(path, map_location='cpu', weights_only=False)
    # prefer 'psds' else 'mean_psds'
    if 'psds' in obj:
        psds = obj['psds']
    elif 'mean_psds' in obj:
        psds = obj['mean_psds']
    else:
        # try to find an array-like putatively PSDs
        found = None
        for k in ['data', 'x', 'spectra']:
            if k in obj:
                found = obj[k]
                break
        if found is None:
            raise KeyError(f"No 'psds' or 'mean_psds' found in {path}. Keys: {list(obj.keys())}")
        psds = found

    # snrs
    snrs = obj['snrs'] if 'snrs' in obj else None

    # ensure numpy or torch arrays
    if isinstance(psds, torch.Tensor):
        psd_arr = psds.cpu().numpy()
    else:
        psd_arr = np.array(psds)

    # reshape handling to ensure psd_arr is 2D before accessing shape[1]
    if psd_arr.ndim == 3 and psd_arr.shape[2] == 1:
        psd_arr = psd_arr.reshape(psd_arr.shape[0], psd_arr.shape[1])
    elif psd_arr.ndim == 1: # If it's a 1D array, make it (1, N) for any N
        psd_arr = psd_arr.reshape(1, -1)

    # Now psd_arr is guaranteed to be 2D. Handle the second dimension to be 192.
    if psd_arr.shape[1] != 192:
        if psd_arr.shape[1] < 192:
            pad = np.zeros((psd_arr.shape[0], 192 - psd_arr.shape[1]))
            psd_arr = np.concatenate([psd_arr, pad], axis=1)
        else: # psd_arr.shape[1] > 192
            psd_arr = psd_arr[:, :192]

    if snrs is not None:
        snr_arr = np.array(snrs).reshape(-1)
        if snr_arr.shape[0] != psd_arr.shape[0]:
            # try to broadcast / trim
            m = min(snr_arr.shape[0], psd_arr.shape[0])
            snr_arr = snr_arr[:m]
            psd_arr = psd_arr[:m, :]
    else:
        snr_arr = None

    return {'psds': psd_arr.astype(np.float32), 'snrs': snr_arr}

# Load all mod files
MOD_DATA = {}
for mod_name, path in MOD_FILES.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing modulation file: {path}")
    print('Loading', mod_name, path)
    MOD_DATA[mod_name] = load_received_file(path)
    print('  -> psds:', MOD_DATA[mod_name]['psds'].shape, ' snrs:', None if MOD_DATA[mod_name]['snrs'] is None else MOD_DATA[mod_name]['snrs'].shape)

# Build PSD_lib using ALL PSDs. For convenience, create list of dicts with 'psd' and 'snr' fields
PSD_LIB = {}
for idx, (mod_name, d) in enumerate(MOD_DATA.items(), start=1):
    psd_matrix = d['psds']  # shape (N,192)
    snr_vector = d['snrs']  # may be None
    entries = []
    for i in range(psd_matrix.shape[0]):
        entry = {'psd': torch.tensor(psd_matrix[i].reshape(192,1), dtype=torch.float32)}
        entry['snr_recorded'] = float(snr_vector[i]) if snr_vector is not None else None
        entries.append(entry)
    PSD_LIB[idx] = entries

# Filter MOD_KEYS to only include those with available PSD entries
MOD_KEYS = [k for k, v in PSD_LIB.items() if len(v) > 0]
MOD_NAME_MAP = {i: name for i, name in enumerate(MOD_DATA.keys(), start=1)}
print('Built PSD_LIB with mods:', MOD_NAME_MAP)
print('Active MOD_KEYS for generation:', MOD_KEYS)

# ---------------------------
# 4) Geometry and assignment (reused from original generator)
# ---------------------------
locat_endpt = {
   0: [-2, 3**.5],
   1: [0, 3**.5],
   2: [2, 3**.5],
   3: [-3, 0],
   4: [-1, 0],
   5: [1, 0],
   6: [3, 0],
   7: [-2, -3**.5],
   8: [0, -3**.5],
   9: [2, -3**.5],
}

locat_centre = {
    0: [-2, 3**.5/3],
    1: [-1, 2/3*3**.5],
    2: [0, 3**.5/3],
    3: [1, 2/3*3**.5],
    4: [2, 3**.5/3],
    5: [-2, -3**.5/3],
    6: [-1, -2/3*3**.5],
    7: [0, -3**.5/3],
    8: [1, -2/3*3**.5],
    9: [2, -3**.5/3],
}

dist_dict = {i: [np.linalg.norm( np.array(locat_endpt[i]) - np.array(locat_centre[j]) ) for j in range(len(locat_centre))] for i in range(len(locat_endpt))}

assign_dict20 = {
    'description': 'The bands allocated to each PU. for 20-band-case',
    'nPU': 10,
    'PU0':[0],
    'PU1':[1, 10],
    'PU2':[2, 11, 14],
    'PU3':[3],
    'PU4':[4, 19],
    'PU5':[5, 13],
    'PU6':[6, 15, 17],
    'PU7':[7, 12, 18],
    'PU8':[8, 16],
    'PU9':[9],
}

class_dir10 = [
    [0,3,4],
    [0,1,4],
    [1,4,5],
    [1,2,5],
    [2,5,6],
    [3,4,7],
    [4,7,8],
    [4,5,8],
    [5,8,9],
    [5,6,9]
]

class_dir20 = [[] for _ in range(len(class_dir10))]
for SU in range(len(class_dir10)):
    for PU in class_dir10[SU]:
        class_dir20[SU].extend(assign_dict20['PU'+str(PU)])

# ---------------------------
# 5) Utility functions: frequency drift, channel gain, compute avg power
# ---------------------------

def fractional_shift(arr, shift):
    """Shift 1D numpy array by a fractional amount using linear interpolation.
    Positive shift moves content to the right.
    """
    n = arr.shape[0]
    x = np.arange(n)
    xp = (x - shift) % n
    return np.interp(x, x, arr[xp.astype(int)])


def apply_freq_drift(psd_tensor, max_frac=FREQ_DRIFT_FRAC):
    """psd_tensor: torch tensor (192,1). Apply a small fractional shift within [-max_frac*192, max_frac*192]."""
    arr = psd_tensor.squeeze(-1).cpu().numpy()
    max_shift = max_frac * arr.shape[0]
    shift = np.random.uniform(-max_shift, max_shift)
    shifted = fractional_shift(arr, shift)
    return torch.tensor(shifted.reshape(192,1), dtype=torch.float32)


def channel_gain(PU, SU, dist_amp=DIST_AMP, alpha=ALPHA, beta=BETA):
    base = (beta * ((dist_amp * dist_dict[PU][SU])**alpha))**(-1)
    # log-normal shadowing (dB)
    shadow_db = np.random.normal(loc=0.0, scale=SHADOW_STD_DB)
    shadow_lin = 10**(shadow_db/10.0)
    return base * shadow_lin


def compute_avg_pw(assign_dict, beta, alpha, nPU, nSU, nch, dist_amp, dist_dictionary):
    total_pw = 0.0
    for SU in range(nSU):
        for PU in range(nPU):
            total_pw += len(assign_dict['PU'+str(PU)]) * (beta * ((dist_amp * dist_dictionary[PU][SU])**alpha))**(-1)
    avg_pw = total_pw / (nSU * nch)
    return avg_pw

# ---------------------------
# 6) Core generator using ALL PSDs and real SNR matching
# ---------------------------

def sample_modulation_for_pu(pu_index, available_mod_keys=MOD_KEYS):
    """Randomly choose a modulation library key for a PU.
       This can be uniform or weighted; currently uniform.
    """
    return random.choice(available_mod_keys)


def sample_psd_from_lib(lib_key):
    """Sample one PSD entry dict from PSD_LIB[lib_key] uniformly at random."""
    entries = PSD_LIB[lib_key]
    idx = random.randrange(len(entries))
    return entries[idx]


def data_generator_realistic(dist_amp, class_dir, dbsize_list, nch, nw, assign_dict, target_snr_list=None,
                             dist_dictionary=dist_dict, use_all_psds=True):
    """Generates data using ALL PSDs, real-SNR matching, random modulation, freq drift and channel fading.
    dbsize_list: list of counts per class (len 2**nPU). For memory reasons we generate class-by-class and store lists.
    target_snr_list: if provided, a global list of SNR dB values to sample from. Otherwise uses recorded SNRs from PSD entries.
    Returns: db_list (list of inputs per sample), label_list (tensor labels)
    """
    nPU = assign_dict['nPU']
    nSU = len(class_dir)
    db = []
    labels = []

    # average power and baseline noise (we will reweight per-sample according to recorded or sampled SNR)
    avg_pw = compute_avg_pw(assign_dict, BETA, ALPHA, nPU, nSU, nch, dist_amp, dist_dictionary)

    M = len(PSD_LIB)

    total_classes = 2**nPU
    for cls in range(total_classes):
        count = dbsize_list[cls]
        for sample_idx in range(count):
            # initialize empty multi-SU input list
            inp = []
            label = torch.zeros(nch)

            # randomly decide modulation per PU for this sample
            pu_mod_keys = [sample_modulation_for_pu(pu, MOD_KEYS) for pu in range(nPU)]

            # choose PSD index for each PU (sample from ALL PSDs)
            pu_psd_entries = [sample_psd_from_lib(pu_mod_keys[pu]) for pu in range(nPU)]

            # For each SU, create multi-channel vector (1 x nw x nch)
            for SU in range(nSU):
                # start with noise: we will add later according to desired SNR per PU
                a = torch.zeros(1, nw, nch)

                # for each PU that is active in this class
                for PU in range(nPU):
                    if cls & (2**PU):
                        # active PU -> get channel gain and PSD
                        gain = channel_gain(PU, SU, dist_amp, ALPHA, BETA)

                        entry = pu_psd_entries[PU]
                        psd = entry['psd']  # (192,1) tensor
                        # apply frequency drift
                        psd_shifted = apply_freq_drift(psd, max_frac=FREQ_DRIFT_FRAC)

                        # compute PSD middle-band average power (indices 64:128)
                        psd_1d = psd_shifted.squeeze(-1)
                        pw = float(psd_1d[64:128].sum().item() / 64.0)
                        if pw == 0:
                            pw = 1.0

                        # determine desired per-PU SNR (dB)
                        if target_snr_list is not None and len(target_snr_list) > 0:
                            chosen_snr_db = float(random.choice(target_snr_list))
                        else:
                            # use recorded entry snr if present, else sample from that modulation's snr distribution
                            if entry.get('snr_recorded') is not None:
                                chosen_snr_db = entry['snr_recorded']
                            else:
                                # fallback: sample randomly from combined snrs across lib_key
                                all_snrs = [e['snr_recorded'] for e in PSD_LIB[pu_mod_keys[PU]] if e['snr_recorded'] is not None]
                                if len(all_snrs) > 0:
                                    chosen_snr_db = float(random.choice(all_snrs))
                                else:
                                    chosen_snr_db = 0.0

                        # Now scale PSD to account for channel gain
                        # signal power after channel: (psd normalized by pw) * gain
                        sig_after_gain = (psd_1d / pw) * gain

                        # noise power required for this PU to achieve chosen_snr_db
                        # SNR (linear) = P_signal / P_noise  -> P_noise = P_signal / SNR_lin
                        P_signal = float(sig_after_gain[64:128].sum().item() / 64.0)
                        if P_signal <= 0:
                            P_signal = 1e-12
                        SNR_lin = 10**(chosen_snr_db / 10.0)
                        P_noise_required = P_signal / SNR_lin

                        # Add the signal into the 3 bands (left:0:64, center:64:128, right:128:192)
                        # center band
                        ch_list = assign_dict['PU'+str(PU)]
                        for ch in ch_list:
                            label[ch] = 1
                            a[0, :, ch].add_(sig_after_gain[64:128])
                            # left leakage
                            if ch > 0:
                                a[0, :, ch-1].add_(sig_after_gain[0:64])
                            # right leakage
                            if ch < (nch - 1):
                                a[0, :, ch+1].add_(sig_after_gain[128:192])

                        # After placing signal, add noise for this PU across channels to reach P_noise_required
                        # We'll add AWGN shaped as white across frequency points: gaussian with variance = P_noise_required
                        noise_std = math.sqrt(max(P_noise_required, 1e-14))
                        noise_tensor = torch.randn(1, nw, nch) * noise_std
                        a.add_(noise_tensor)

                # if no PU active, still add small baseline noise
                if (cls == 0) or (not (cls & ((1<<nPU)-1))):
                    a.add_(torch.randn_like(a) * 1e-6)

                # finally, store absolute value (like original generator used abs)
                inp.append(torch.abs(a))

            db.append(deepcopy(inp))
            labels.append(deepcopy(label))

    return db, labels

# ---------------------------
# 7) Prepare dbsize lists and run generation
# ---------------------------
TOTAL_CLASSES = 2**NPU
DBSIZE_TRAIN = [VOL_TRAIN] * TOTAL_CLASSES
DBSIZE_TEST = [VOL_TEST] * TOTAL_CLASSES

# Optionally, you can provide a global SNR list to sample from (uncomment to use)
GLOBAL_SNR_LIST = [0.0, -2.0, -4.0, -6.0, -8.0, -10.0, -12.0, -14, -16] # Example: set to a single SNR value

print('Starting generation: classes=', TOTAL_CLASSES, 'train per class=', VOL_TRAIN, 'test per class=', VOL_TEST)

# new: helper to make filename-friendly snr string and a small wrapper to run+save
def _format_snr_for_filename(snr):
    # represent ints without decimal, otherwise replace dot to 'p' to avoid filesystem dots
    try:
        f = float(snr)
    except Exception:
        return str(snr)
    if abs(f - int(f)) < 1e-6:
        return str(int(f))
    return str(snr).replace('.', 'p')

def _run_and_save(target_snr_list, out_path):
    print(f'Generating dataset for target_snr_list={target_snr_list} -> {out_path}')
    datas_tr, labels_tr = data_generator_realistic(DIST_AMP, class_dir20, DBSIZE_TRAIN, NCH, NW, assign_dict20,
                                                   target_snr_list=target_snr_list, dist_dictionary=dist_dict)
    datas_te, labels_te = data_generator_realistic(DIST_AMP, class_dir20, DBSIZE_TEST, NCH, NW, assign_dict20,
                                                   target_snr_list=target_snr_list, dist_dictionary=dist_dict)
    OUT_META = {
        'Description': f'RealisticReceivedPSD_Generated_{TIME_TAG}',
        'training data list': datas_tr,
        'training label list': labels_tr,
        'testing data list': datas_te,
        'testing label list': labels_te,
        'time': TIME_TAG,
        'classdir': class_dir20,
    }
    torch.save(OUT_META, out_path)
    print('Saved.', out_path)

# Decide behavior:
if GLOBAL_SNR_LIST is None:
    # Use recorded SNRs (existing behavior), keep time-tagged filename
    out_path = os.path.join(OUT_DIR, f'ReceivedData_realistic_vol{VOL_TRAIN}_time{TIME_TAG}_SNR_recorded.pth')
    _run_and_save(None, out_path)

else:
    # If GLOBAL_SNR_LIST contains multiple values, produce one file per SNR value.
    # If it contains a single value, also produce one file for that SNR.
    if len(GLOBAL_SNR_LIST) == 0:
        raise ValueError('GLOBAL_SNR_LIST is empty.')
    for snr in GLOBAL_SNR_LIST:
        snr_tag = _format_snr_for_filename(snr)
        out_fname = f'Data_SNR{snr_tag}vol{VOL_TRAIN}.pth'  # e.g. Data_SNR-4vol20.pth
        out_path = os.path.join(OUT_DIR, out_fname)
        _run_and_save([snr], out_path)
