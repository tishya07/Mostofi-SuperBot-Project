# cleaned and updated version of spectrogram.py
import spec_functions as sf


class SpectrogramGenerator:
    def __init__(self, CSI_data):
        self.fs = 500 # Sampling rate of our data fs=1/detla*t
        self.N_packets, N_streams = CSI_data.shape# from CSI we get two columns, smaples=packets and subcarriers=streams
        self.n_components=20 #number of principal components to use
        self.f_min=2
        self.f_max=100
        self.num_skip_pcs=1 #number of pcs to skip

    def prep_CSI_data(self, raw_CSI_data):
        """
        Given raw CSI data, process and return PCA-projected data (time series data)

        CSI_data : ndarray
            Raw CSI data (N_packets, N_streams)
        """
        time_total = np.arange(self.N_packets) / self.fs  # Time vector: each data stamp is taken at a frequncy of 500Hz (assumed from the file being labled 500)
        duration = time_total[-1] # gets last index meanign total duration of the signal
        CSI_mag_squared = np.abs(raw_CSI_data[:,:]) #**2 #this says take the magnitude of all samples in all columns(magnitude all subcarriers) and square
        signal_clean = CSI_mag_squared - np.mean(CSI_mag_squared, axis=0)# remove DC component (might need changing if we want to add realtime data)
        signal_filtered = bandpass_filter(signal_clean, self.fs, low=self.f_min, high=self.f_max, order=4)

        X_PCA, eigvals, eigvecs = compute_pca_components(signal_filtered,
                                                                n_components=self.n_components,
                                                                skip_first=self.num_skip_pcs,
                                                                whiten=False)
        return X_PCA


# raw_CSI_data must be loaded from .mat file
spec = SpectrogramGenerator(raw_CSI_data)
csi_post_pca = spec.prep_CSI_data(raw_CSI_data)
f, t, mag = sf.compute_STFT(csi_post_pca, spec.fs, T_win=0.4)
STFT_data = sf.process_stft_results(mag, f)


sf.plot_spectrogram(f, t, STFT_data,
                 title="STFT",
                 f_max=100,
                 cmap='jet',#jet, viridis
                 dB=False,

                 figsize=(14,6))
