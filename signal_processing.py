import numpy as np
import pywt
from scipy.signal import welch
from config import Config

class SignalProcessor:
    def __init__(self):
        self.config = Config()
    
    def wavelet_packet_decomposition(self, signal):
        wp = pywt.WaveletPacket(signal, wavelet=self.config.WAVELET_TYPE, maxlevel=self.config.WAVELET_LEVELS)
        
        subbands = []
        for node in wp.get_level(self.config.WAVELET_LEVELS, 'natural'):
            subbands.append(node.data)
        
        return subbands
    
    def compute_sliding_window_energy(self, subband):
        window_size = self.config.WINDOW_SIZE
        overlap_step = self.config.OVERLAP_STEP
        
        if len(subband) < window_size:
            return [np.sum(subband**2)]
        
        energies = []
        start = 0
        while start + window_size <= len(subband):
            window_data = subband[start:start + window_size]
            energy = np.sum(window_data**2)
            energies.append(energy)
            start += overlap_step
        
        return energies
    
    def compute_center_frequency(self, subband, fs=1000):
        if len(subband) < 2:
            return 0.0
        
        freqs, psd = welch(subband, fs=fs, nperseg=min(256, len(subband)))
        
        if np.sum(psd) == 0:
            return 0.0
        
        center_freq = np.sum(freqs * psd) / np.sum(psd)
        return center_freq
    
    def process_signal(self, signal):
        subbands = self.wavelet_packet_decomposition(signal)
        
        energy_sequences = []
        center_frequencies = []
        
        for subband in subbands:
            energies = self.compute_sliding_window_energy(subband)
            center_freq = self.compute_center_frequency(subband)
            
            energy_sequences.append(energies)
            center_frequencies.append(center_freq)
        
        return energy_sequences, center_frequencies 