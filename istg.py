import numpy as np
from scipy.spatial.distance import cosine
from config import Config

class ISTG:
    def __init__(self):
        self.config = Config()
    
    def compute_time_similarity(self, energy_seq1, energy_seq2):
        if len(energy_seq1) == 0 or len(energy_seq2) == 0:
            return 0.0
        
        max_len = max(len(energy_seq1), len(energy_seq2))
        
        padded_seq1 = np.pad(energy_seq1, (0, max_len - len(energy_seq1)), 'constant')
        padded_seq2 = np.pad(energy_seq2, (0, max_len - len(energy_seq2)), 'constant')
        
        norm1 = np.linalg.norm(padded_seq1)
        norm2 = np.linalg.norm(padded_seq2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(padded_seq1, padded_seq2) / (norm1 * norm2)
    
    def compute_freq_similarity(self, freq1, freq2):
        return 1.0 / (1.0 + abs(freq1 - freq2))
    
    def compute_joint_similarity(self, energy_seq1, energy_seq2, freq1, freq2):
        time_sim = self.compute_time_similarity(energy_seq1, energy_seq2)
        freq_sim = self.compute_freq_similarity(freq1, freq2)
        
        joint_sim = self.config.ALPHA * freq_sim + (1 - self.config.ALPHA) * time_sim
        return joint_sim
    
    def build_adjacency_matrix(self, energy_sequences, center_frequencies):
        num_bands = len(energy_sequences)
        adjacency_matrix = np.zeros((num_bands, num_bands))
        
        for i in range(num_bands):
            for j in range(num_bands):
                if i != j:
                    similarity = self.compute_joint_similarity(
                        energy_sequences[i], energy_sequences[j],
                        center_frequencies[i], center_frequencies[j]
                    )
                    
                    if similarity > self.config.SIMILARITY_THRESHOLD:
                        adjacency_matrix[i, j] = similarity
        
        return adjacency_matrix
    
    def compute_laplacian_eigenvalues(self, adjacency_matrix):
        degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
        laplacian_matrix = degree_matrix - adjacency_matrix
        
        eigenvalues, _ = np.linalg.eigh(laplacian_matrix)
        eigenvalues = np.sort(eigenvalues)
        
        return eigenvalues
    
    def build_istg(self, energy_sequences, center_frequencies):
        adjacency_matrix = self.build_adjacency_matrix(energy_sequences, center_frequencies)
        eigenvalues = self.compute_laplacian_eigenvalues(adjacency_matrix)
        
        return eigenvalues, adjacency_matrix 