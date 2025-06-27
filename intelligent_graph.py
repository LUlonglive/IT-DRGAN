import numpy as np
import torch
from torch_geometric.data import Data
from signal_processing import SignalProcessor
from istg import ISTG
from der import DER
from config import Config

class IntelligentGraphBuilder:
    def __init__(self):
        self.config = Config()
        self.signal_processor = SignalProcessor()
        self.istg_builder = ISTG()
        self.der = DER()
    
    def extract_features_from_signals(self, signals):
        features = []
        
        for signal in signals:
            energy_sequences, center_frequencies = self.signal_processor.process_signal(signal)
            eigenvalues, _ = self.istg_builder.build_istg(energy_sequences, center_frequencies)
            features.append(eigenvalues)
        
        max_length = max(len(f) for f in features)
        padded_features = []
        for feature in features:
            padded = np.pad(feature, (0, max_length - len(feature)), 'constant')
            padded_features.append(padded)
        
        return np.array(padded_features)
    
    def build_intelligent_graph(self, signals, labels):
        features = self.extract_features_from_signals(signals)
        
        adjacency_matrix = self.der.learn_intelligent_graph(features)
        
        edge_index = self.adjacency_to_edge_index(adjacency_matrix)
        
        x = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)
        
        graph_data = Data(x=x, edge_index=edge_index, y=y)
        
        return graph_data, adjacency_matrix
    
    def adjacency_to_edge_index(self, adjacency_matrix):
        edge_indices = np.nonzero(adjacency_matrix)
        edge_index = torch.tensor(np.vstack(edge_indices), dtype=torch.long)
        return edge_index 