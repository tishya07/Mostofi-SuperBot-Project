import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.signal import stft, get_window
from scipy.ndimage import gaussian_filter

# Config
path = r"c:/Users/ptv57/Downloads/data_500.mat"  
fs = 500                        # packets/s (from Xmodal assuming it's close)
static_avg_secs = 4.0            # long-term averaging window to estimate static offset
chunk_secs = 1.0                 # length of analysis window in seconds
discard_first_pc_when_reconstructing = True  # recommended in reference paper


#Load CSI (.mat) and find the array
mat = sio.loadmat(path)
keys = [k for k in mat.keys() if not k.startswith("__")]
if not keys:
    raise ValueError("No data variables found in the .mat file.")
var_name = keys[0]
X = mat[var_name]  # expected shape (T, N_streams), dtype complex128
print(f"Loaded '{var_name}' with shape {X.shape} and dtype {X.dtype}")

# Sanity checks
if X.ndim != 2:
    raise ValueError("Expected a 2-D array (time x streams).")

T, N = X.shape

#Use magnitudes to kill CFO sensitivity (ignore phase)
# s_b(t) -> |s_b(t)| per stream/subcarrier
A = np.abs(X).astype(np.float64)   # shape (T, N)


# Remove static component (long-term mean over ~4 s per stream)
L_static = int(round(static_avg_secs * fs))
if L_static < T:
    static_offset = A[:L_static].mean(axis=0)  # mean for each stream over first 4 s
else:
    static_offset = A.mean(axis=0)             # fallback if record < 4 s

A_dc = A - static_offset  # remove static paths per stream

#Chop into 1-s non-overlapping chunks to form H matrices
L = int(round(chunk_secs * fs))  # samples per chunk
n_chunks = T // L                 # drop tail if not full second
if n_chunks == 0:
    raise ValueError("Recording too short for the chosen chunk length.")

# Pre-allocate per-chunk explained variance (eigenvalue ratios)
explained_ratios = np.zeros((n_chunks, N), dtype=np.float64)

pcs_90 = np.zeros(n_chunks, dtype=int)

pcs_95 = np.zeros(n_chunks, dtype=int)

def eigen_explained_variance(H):
    """
    Given H (L x N), column-centered, compute R = H^T H,
    eigendecompose, and return sorted eigenvalue ratios (descending).
    """
    # Correlation matrix (N x N)
    R = H.T @ H
    # Symmetric => use eigh; returns ascending eigenvalues
    w, v = np.linalg.eigh(R)
    # Sort descending
    idx = np.argsort(w)[::-1]
    w = w[idx]
    v = v[:, idx]
    # Explained variance ratio = eigenvalue / sum(eigenvalues)
    w_sum = w.sum()
    # Handle degenerate case (all zeros)
    if w_sum <= 0:
        evr = np.zeros_like(w)
    else:
        evr = w / w_sum
    return evr, v


# Per-chunk PCA via eigenanalysis of H^T H

for k in range(n_chunks):
    # Take 1-s slice
    H = A_dc[k*L:(k+1)*L, :]              # shape (L, N)
    # Column-center H (mean 0 per stream within the chunk)
    H = H - H.mean(axis=0, keepdims=True)

    # Eigen on R = H^T H -> explained variance ratios
    evr, eigvecs = eigen_explained_variance(H)
    explained_ratios[k, :] = evr

    # PCs needed to hit 90% / 95%
    cume = np.cumsum(evr)
    pcs_90[k] = int(np.searchsorted(cume, 0.90, side="left") + 1)
    pcs_95[k] = int(np.searchsorted(cume, 0.95, side="left") + 1)

    
    

# parameters
Twin_sec = 0.40          # STFT window length in seconds (Xmodal)
hop_sec  = 0.004         # STFT hop (shift) in seconds (Xmodal = .004)
f_lo, f_hi = 15.0, 125.0 # frequency band to display (Xmodal)
npcs_to_average = 20     # how many PC waveforms to include in the average 16 = 98%
drop_first_pc = False    # discard 1st PC (recommended in CARM paper)
normalize_each_pc = True # normalize each PC spectrogram before averaging
nfft = 1024             # FFT size for STFT (>= nperseg); adjust if you want finer freq grid

# build PCs on the full recording
# A_dc magnitudes after static offset removal from earlier code
# column-center across the recording so PCA sees zero-mean features
A0 = A_dc - A_dc.mean(axis=0, keepdims=True)

# correlation matrix R = H^T H
R_full = A0.T @ A0

# symmetric eigendecomposition; eigh returns ascending eigenvalues
w_full, Q_full = np.linalg.eigh(R_full)

# sort eigenvalues/eigenvectors in descending variance order
idx = np.argsort(w_full)[::-1]
w_full = w_full[idx]
Q_full = Q_full[:, idx]

# project the entire time series onto PC directions to get PC waveforms: Hpc_all = A0 Q
Hpc_all = A0 @ Q_full   # shape (T, N); column i is time series for PC i (h_i)

# choose which PCs to use for spectrogram averaging
start_idx = 1 if drop_first_pc else 0              # skip PC1 if requested
stop_idx  = 20            # take next npcs_to_average
Hpcs_sel  = Hpc_all[:, start_idx:stop_idx]         # shape (T, npcs_to_average)

# STFT settings
nperseg = int(round(Twin_sec * fs))                # samples per STFT window
hop     = int(round(hop_sec * fs))                 # hop in samples
noverlap = nperseg - hop                           # overlap = window - hop
win = get_window('hann', nperseg, fftbins=True)    # Hann window

# compute and average spectrograms over selected PCs
S_accum = None
t_ref, f_ref = None, None  # to keep common time/freq axes

for i in range(Hpcs_sel.shape[1]):
    x = Hpcs_sel[:, i]                             # time series of PC i
    #STFT
    f, t, Z = stft(x, fs=fs, window=win, nperseg=nperseg,
                   noverlap=noverlap, nfft=nfft, boundary=None, padded=False)
    mag = np.abs(Z)

    # normalize per PC so averaging isn’t dominated by one PC
    if normalize_each_pc:
        m = np.max(mag) + 1e-12
        mag = mag / m

    # initialize accumulator and references on first iteration
    if S_accum is None:
        S_accum = np.zeros_like(mag, dtype=np.float64)
        f_ref, t_ref = f, t

    # sanity: make sure shapes match if using multiple PCs
    if mag.shape != S_accum.shape:
        raise ValueError("Inconsistent STFT shape across PCs. "
                         "Check fs/nperseg/hop/nfft and boundary options.")

    # accumulate
    S_accum += mag

# average across PCs
S_avg = S_accum / Hpcs_sel.shape[1]

# trim to the frequency band (15–125 Hz)
band = (f_ref >= f_lo) & (f_ref <= f_hi)           # boolean mask for chosen band
f_band = f_ref[band]
S_band = S_avg[band, :]


'''# # Normalize per time slice after noise thresholding
S_sum = np.sum(S_band, axis = 0)
S_norm = S_band / (S_sum + 1e-12)
## for global normalization
# S_norm = S_thresh / (np.max(S_thresh) + 1e-12)

def noise_floor_adjustment(S, f_band, noise_threshold = 70):
    # mean magnitude where the frequencies are greater than a threshold value
    noise_floor = np.mean(S[f_band > noise_threshold,:])
    # Where the magnitudes are greater than the noise_floor
    # S_threshold = np.where(S >= noise_floor, S, 0)
    S_threshold = S - noise_floor
    S_threshold = np.where(S_threshold >= 0, S_threshold, 0)

    return S_threshold



def plot_spectrogram(spectrogram, title):
    plt.figure(figsize=(9,4.5))
    extent = [t_ref[0], t_ref[-1], f_band[0], f_band[-1]]
    plt.imshow(spectrogram, aspect='auto', origin='lower', extent=extent, interpolation='nearest')
    plt.colorbar(label='Magnitude linear')
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


S_thresh = noise_floor_adjustment(S_norm, f_band)

plot_spectrogram(S_avg, "Average of the PCA spectrograms")
plot_spectrogram(S_band, "Spectrogram with frequency band of 15-125 Hz")
plot_spectrogram(S_norm, "Spectrogram normalized per time stamp")
plot_spectrogram(S_thresh, "Spectrogram with noise floor threshold (70 Hz) after normalization")'''

# Trying to bring out the turn portion of spectrogram by subtracting global noise 
# noise floor and bounding per-frame gain

# Remove small global baseline to kill very low-level noise
floor_global = np.percentile(S_band, 10)   # 10th percentile across ALL freq+time
S_nf = S_band - floor_global
S_nf = np.maximum(S_nf, 0.0)

# For each time frame estimate a "level" 
frame_level = np.percentile(S_nf, 95, axis=0)   # 95th percentile of that column

# Reference level = median over nonzero frames
valid = frame_level > 0
if np.any(valid):
    ref_level = np.median(frame_level[valid])
else:
    ref_level = 1.0

eps = 1e-12
gains = ref_level / (frame_level + eps)   

# Clip gains so we don't blow up noise
gains = np.clip(gains, 0.5, 3.0)         

# Apply gains per column
S_eq = S_nf * gains[np.newaxis, :]  

# Final global normalization for ploting 
S_norm = S_eq / (S_eq.max() + 1e-12)

plt.figure(figsize=(9,4.5))
extent = [t_ref[0], t_ref[-1], f_band[0], f_band[-1]]

plt.imshow(S_norm, aspect='auto', origin='lower',
           extent=extent, interpolation='nearest')
plt.colorbar(label='Magnitude ')
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title(f"Averaged STFT of {npcs_to_average} PCA waveforms "
          f"({'PC2..' if drop_first_pc else 'PC1..'}PC{stop_idx})\n"
          f"Hann, {Twin_sec:.3f}s window, {hop_sec*1000:.0f} ms hop")
plt.tight_layout()
plt.show()

