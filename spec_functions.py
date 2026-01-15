import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.special import hermite, factorial
import scipi.io as sio
from scipy.signal import stft, get_window
from scipy.ndimage import gaussian_filter1d

super().__init__()

def compute_STFT(signal_in, fs, T_win=0.4):
    """
    Computes STFT for each column of CSI magnitude.

    Parameters:
        signal_in : np.ndarray
            2D CSI data (time x subcarriers) or 1D array (single time series).
        fs : float
            Sampling rate (Hz)
        T_win : float
            Window length (seconds)
        T_overlap : float
            Overlap length (seconds)
    Returns:
        freq : list of arrays
            Frequency arrays for each column
        time : list of arrays
            Time arrays for each column
        mag : list of 2D arrays
            STFT magnitude arrays for each column
    """
    # Determine if the original input was a single time series (1D or 2D with one column)
    is_single_time_series = (signal_in.ndim == 1) or (signal_in.ndim == 2 and signal_in.shape[1] == 1)

    # Ensure signal_in is always (N_samples, N_components) for consistent processing
    if signal_in.ndim == 1:
        signal_in_processed = signal_in.reshape(-1, 1)
    elif signal_in.ndim == 2 and signal_in.shape[0] == 1 and signal_in.shape[1] > 1:
        # If it's a row vector (1, N), transpose to (N, 1) to treat as N_samples, 1_component
        signal_in_processed = signal_in.T
    else:
        signal_in_processed = signal_in

    T_overlap = T_win * 0.99

    nperseg = int(T_win * fs)
    noverlap = int(T_overlap * fs)

    # Initialize as empty lists
    freq = []
    time = []
    mag = []

    for i in range(signal_in_processed.shape[1]):
        # Compute STFT for this subcarrier/component
        f, t, Zxx = signal.stft( #check overlap vs how much you step
            signal_in_processed[:, i],
            fs=fs,
            window='hann', #gaussian
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=False,
            return_onesided=True
        )
        # Append to lists instead of indexing
        freq.append(f)
        time.append(t)
        mag.append(np.abs(Zxx))

    # Return based on whether the original input was a single time series
    # f and t are compressed because they stay the same for all inputs as they are fucntions of fs Twin and t_overlap
    if is_single_time_series:
        return np.array(freq[0]), np.array(time[0]), np.array(mag[0])
    else:
        return np.array(freq[0]), np.array(time[0]), np.array(mag)


def plot_spectrogram(f, t, S,
                     title="Spectrogram",
                     f_max=100,
                     cmap='jet',
                     dB=True,
                     figsize=(14,6)):

    """
    Plots a spectrogram given frequency, time, and magnitude values.

    Parameters:
        f : 1D array
            Frequencies (Hz)
        t : 1D array
            Time stamps (s)
        S : 2D array
            Magnitude or power matrix with shape (len(f), len(t))
        title : str
            Plot title
        f_max : float
            Maximum frequency to show (Hz)
        cmap : str
            Matplotlib colormap
        dB : bool
            If True, convert magnitude to dB
        vmin_percentile : float
            Lower percentile for dynamic range scaling
        vmax_percentile : float
            Upper percentile for dynamic range scaling
        figsize : tuple
            Figure size
    """

    # Convert to dB if needed
    if dB:
        S_plot = 10 * np.log10(S + 1e-12)
    else:
        S_plot = S

    # Dynamic range scaling (prevents overly bright outliers)
    #vmin = np.percentile(S_plot, vmin_percentile)
    #vmax = np.percentile(S_plot, vmax_percentile)

    # Plot
    plt.figure(figsize=figsize)
    plt.pcolormesh(t, f, S_plot,
                   shading='gouraud',
                   cmap=cmap)

    plt.title(title, fontsize=14)
    plt.xlabel("Time [s]", fontsize=12)
    plt.ylabel("Frequency [Hz]", fontsize=12)

    # Limit frequency axis
    if f_max is not None:
        plt.ylim([0, f_max])

    cbar = plt.colorbar()
    cbar.set_label("Power (dB)" if dB else "Magnitude", fontsize=11)

    plt.tight_layout()
    plt.show()

def compute_pca_components(signal_clean, n_components=80, skip_first=1, whiten=False):
    """
    Compute PCA on CSI data and project onto principal components.

    Parameters:
    -----------
    signal_clean : ndarray
        Preprocessed CSI data, shape (T, N) where:
        - T = number of time samples
        - N = number of subcarriers/streams
    n_components : int
        Number of principal components to extract (default: 80)
    skip_first : int
        Number of first PCs to skip (default: 1, skips PC1 which contains noise)
    whiten : bool
        If True, apply whitening to normalize variance across PCs (default: False)
        [TODO]: delete variable entirely
    Returns:
    --------
    X_proj : ndarray
        Projected data onto selected PCs, shape (T, n_components)
        If whiten=True, this is whitened; otherwise raw projections
    eigvals : ndarray
        All eigenvalues sorted descending, shape (N,)
    eigvecs : ndarray
        All eigenvectors sorted by eigenvalue, shape (N, N)
    """

    # Step 1: Compute covariance matrix
    T, N = signal_clean.shape
    Cov = (signal_clean.T @ signal_clean) / (T - 1)

    # Step 2: Compute eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eig(Cov)

    # Step 3: Sort eigenvalues (descending) and reorder eigenvectors
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]  # columns are eigenvectors

    # Step 4: Select PCs from skip_first to (skip_first + n_components)
    V_k = eigvecs[:, skip_first:skip_first + n_components]  # shape (N, n_components)

    # Step 5: Project data onto selected PCs
    X_proj = signal_clean @ V_k  # shape (T, n_components)

    # Step 6: Optional whitening
    if whiten:
        # Get eigenvalues corresponding to selected PCs
        lambda_k = eigvals[skip_first:skip_first + n_components]  # shape (n_components,)
        whitening_scale = 1.0 / np.sqrt(lambda_k)  # shape (n_components,)

        # Apply whitening (broadcast over time dimension)
        X_proj = X_proj * whitening_scale[np.newaxis, :]

    # Print diagnostic info
    total_variance = np.sum(eigvals)
    selected_variance = np.sum(eigvals[skip_first:skip_first + n_components])
    print(f"PCA Info:")
    print(f"  Input shape: {signal_clean.shape}")
    print(f"  Using PCs {skip_first+1} to {skip_first+n_components}")
    print(f"  Variance explained: {selected_variance/total_variance*100:.2f}%")
    print(f"  Whitening: {'ON' if whiten else 'OFF'}")
    print(f"  Output shape: {X_proj.shape}")

    return X_proj, eigvals, eigvecs

def adaptive_noise_floor_per_pc(S, f_band, f1, f2):
    """
    Adaptive noise floor removal for spectrograms.

    Parameters:
    -----------
    S : ndarray
        Spectrogram(s) with shape:
        - (n_freqs, n_times) for single spectrogram, OR
        - (n_pcs, n_freqs, n_times) for multiple PC spectrograms
    f_band : ndarray
        Frequency values, shape (n_freqs,)
    noise_threshold : float
        Frequency (Hz) above which to estimate noise floor

    Returns:
    --------
    S_clean : ndarray
        Cleaned spectrogram, same shape as input S
    """
    # Handle both 2D and 3D inputs
    if S.ndim == 2:
        # Single spectrogram: (n_freqs, n_times)
        # Add dummy dimension to treat as single PC
        S = S[np.newaxis, :, :]
        squeeze_output = True
    elif S.ndim == 3:
        # Multiple PC spectrograms: (n_pcs, n_freqs, n_times)
        squeeze_output = False
    else:
        raise ValueError(f"S must be 2D or 3D, got shape {S.shape}")

    n_pcs, n_freqs, n_times = S.shape
    S_clean = np.zeros_like(S)

    # Select frequencies in the noise band f1 → f2
    noise_mask = (f_band >= f1) & (f_band <= f2)

    for pc in range(n_pcs):
        # Extract this PC's spectrogram
        S_pc = S[pc]  # shape (n_freqs, n_times)

        # Noise region for this PC
        noise_region = S_pc[noise_mask, :]

        # Compute noise floor per time bin
        noise_floor = np.mean(noise_region, axis=0)  # shape (n_times,)

        # Subtract noise floor
        S_clean_pc = S_pc - noise_floor[np.newaxis, :]

        # Clip negative values
        S_clean_pc = np.maximum(S_clean_pc, 0)

        # Store result
        S_clean[pc] = S_clean_pc

    # Remove dummy dimension if input was 2D
    if squeeze_output:
        S_clean = S_clean[0]

    return S_clean

def normalize_by_sum_per_time(mag_pca):
    """
    Normalize by sum of all frequencies per time step (paper's method).

    Parameters:
        mag_pca: (n_pcs, n_freqs, n_times) array of PC spectrograms

    Returns:
        mag_normalized: Same shape, normalized per time step
    """
    n_pcs, n_freqs, n_times = mag_pca.shape
    mag_normalized = np.zeros_like(mag_pca)

    for pc in range(n_pcs):
        # Sum over all frequencies at each time step
        sum_per_t = np.sum(mag_pca[pc], axis=0, keepdims=True)  # Shape: (1, n_times)

        # Normalize (creates probability distribution over frequencies)
        mag_normalized[pc] = mag_pca[pc] / sum_per_t

    return mag_normalized


def bandpass_filter(data, fs, low=0.3, high=60, order=4):
    """
    Apply a bandpass filter to raw CSI data in the frequency domain

    """
    from scipy.signal import butter, filtfilt
    b, a = butter(order, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, data, axis=0)

def hermite_function(n, t):
    """
    Generate the nth Hermite function.

    Parameters:
    -----------
    n : int
        Order (0, 1, 2, ...)
    t : array
        Time values

    Returns:
    --------
    chi_n : array
        Hermite function values
    """
    # Get Hermite polynomial
    H_n = hermite(n)

    # Normalization constant
    norm = 1.0 / np.sqrt(2**n * factorial(n) * np.sqrt(np.pi))

    # Hermite function = polynomial * Gaussian
    chi_n = norm * H_n(t) * np.exp(-t**2 / 2)

    return chi_n


def stft_with_hermite_window(signal_in, fs, T_win=0.4, hermite_order=0):
    """
    Compute STFT using a Hermite function as the window.

    This is just regular STFT but with a Hermite function window
    instead of Hann/Hamming/etc.

    Parameters:
    -----------
    signal_in : array
        1D or 2D array (time) or (time x components)
    fs : float
        Sampling rate (Hz)
    T_win : float
        Window length (seconds)
    hermite_order : int
        Which Hermite function to use (0, 1, 2, ...)
        0 = Gaussian-like
        1 = First Hermite
        2 = Second Hermite, etc.

    Returns:
    --------
    freq : array
        Frequency array
    time : array
        Time array
    mag : array
        Magnitude spectrogram
        2D if input is 1D, 3D if input is 2D
    """
    # Handle input dimensions
    is_single_time_series = (signal_in.ndim == 1) or (signal_in.ndim == 2 and signal_in.shape[1] == 1)

    if signal_in.ndim == 1:
        signal_in_processed = signal_in.reshape(-1, 1)
    elif signal_in.ndim == 2 and signal_in.shape[0] == 1 and signal_in.shape[1] > 1:
        signal_in_processed = signal_in.T
    else:
        signal_in_processed = signal_in

    # Window parameters
    T_overlap = T_win * 0.99
    nperseg = int(T_win * fs)
    noverlap = int(T_overlap * fs)

    # Create Hermite window
    t_window = np.linspace(-T_win/2, T_win/2, nperseg)
    hermite_window = hermite_function(hermite_order, t_window)

    # Initialize output lists
    freq = []
    time = []
    mag = []

    # Process each component
    for i in range(signal_in_processed.shape[1]):
        f, t, Zxx = signal.stft(
            signal_in_processed[:, i],
            fs=fs,
            window=hermite_window,  # Use Hermite window instead of 'hann'
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=False,
            return_onesided=True
        )
        freq.append(f)
        time.append(t)
        mag.append(np.abs(Zxx))

    # Return in same format as csi_stft
    if is_single_time_series:
        return np.array(freq[0]), np.array(time[0]), np.array(mag[0])
    else:
        return np.array(freq[0]), np.array(time[0]), np.array(mag)




def process_stft_results(mag, f):
    """
    Normalized magnitiude data of time-frequency- to be plotted as spectrogram

    mag : ndarray
        Spectrogram magnitude data (n_pcs, n_freqs, n_times)
    f : ndarray
        Frequency values (n_freqs,) 
    """
    mag_norm = normalize_by_sum_per_time(mag)
    mag_avg  = np.mean(mag_norm, axis=0)
    mag_nf   = adaptive_noise_floor_per_pc(mag_avg, f, 60, 80)
    return mag_nf


