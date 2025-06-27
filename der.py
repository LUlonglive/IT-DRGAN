import numpy as np
from sklearn.neighbors import NearestNeighbors
from config import Config

class DER:
    def __init__(self):
        self.config = Config()
    
    def compute_distance_matrix(self, features):
        n_samples = features.shape[0]
        distance_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(n_samples):
                distance_matrix[i, j] = np.linalg.norm(features[i] - features[j])
        
        return distance_matrix
    
    def gaussian_kernel(self, distance_matrix, sigma_g):
        similarity_matrix = np.exp(-distance_matrix**2 / (2 * sigma_g**2))
        return similarity_matrix
    
    def build_initial_adjacency_matrix(self, features):
        distance_matrix = self.compute_distance_matrix(features)
        
        similarity_matrix = self.gaussian_kernel(distance_matrix, self.config.SIGMA_G)
        
        n_samples = features.shape[0]
        adjacency_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            distances_from_i = distance_matrix[i, :]
            k_nearest_indices = np.argsort(distances_from_i)[1:self.config.K_NEIGHBORS+1]
            
            for j in k_nearest_indices:
                adjacency_matrix[i, j] = similarity_matrix[i, j]
        
        adjacency_matrix = (adjacency_matrix + adjacency_matrix.T) / 2
        
        return adjacency_matrix
    
    def update_adjacency_matrix(self, adjacency_matrix, features, iteration):
        n_samples = adjacency_matrix.shape[0]
        new_adjacency = np.zeros_like(adjacency_matrix)
        
        distance_matrix = self.compute_distance_matrix(features)
        current_similarity = self.gaussian_kernel(distance_matrix, self.config.SIGMA_G)
        
        for i in range(n_samples):
            for j in range(n_samples):
                if i != j:
                    degree_regularization = self.config.BETA * np.log(np.sum(adjacency_matrix[i, :]) + 1)
                    
                    similarity_term = self.config.GAMMA * current_similarity[i, j]
                    
                    new_adjacency[i, j] = degree_regularization + similarity_term
                else:
                    new_adjacency[i, j] = 0
        
        if self.config.MAINTAIN_SYMMETRY:
            new_adjacency = (new_adjacency + new_adjacency.T) / 2
        
        return new_adjacency
    
    def learn_intelligent_graph(self, features):
        print(f"Learning Intelligent Graph with DER...")
        print(f"Features shape: {features.shape}")
        print(f"Parameters: K={self.config.K_NEIGHBORS}, σ_g={self.config.SIGMA_G}, β={self.config.BETA}, γ={self.config.GAMMA}")
        
        adjacency_matrix = self.build_initial_adjacency_matrix(features)
        print(f"Initial adjacency matrix constructed with {np.count_nonzero(adjacency_matrix)} non-zero elements")
        
        for iteration in range(self.config.MAX_ITERATIONS):
            new_adjacency = self.update_adjacency_matrix(adjacency_matrix, features, iteration)
            
            change = np.linalg.norm(new_adjacency - adjacency_matrix, 'fro')
            
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}: Frobenius norm change = {change:.6f}")
            
            if change < self.config.CONVERGENCE_THRESHOLD:
                print(f"Converged at iteration {iteration + 1} with change = {change:.6f}")
                break
            
            adjacency_matrix = new_adjacency
        
        print(f"Final adjacency matrix has {np.count_nonzero(adjacency_matrix)} non-zero elements")
        return adjacency_matrix 