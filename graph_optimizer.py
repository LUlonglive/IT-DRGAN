import numpy as np
import torch
import pickle
from torch_geometric.data import Data
from data_loader import DataLoader
from signal_processing import SignalProcessor
from istg import ISTG
from der import DER
from config import Config

class GraphOptimizer:
    def __init__(self):
        self.config = Config()
        self.data_loader = DataLoader()
        self.signal_processor = SignalProcessor()
        self.istg_builder = ISTG()
        self.der = DER()
    
    def extract_features_from_signals(self, signals):
        print("Extracting features from signals...")
        features = []
        
        for i, signal in enumerate(signals):
            if (i + 1) % 100 == 0:
                print(f"Processing signal {i+1}/{len(signals)}")
            
            try:
                energy_sequences, center_frequencies = self.signal_processor.process_signal(signal)
                eigenvalues, _ = self.istg_builder.build_istg(energy_sequences, center_frequencies)
                features.append(eigenvalues)
            except Exception as e:
                print(f"Error processing signal {i}: {e}")
                features.append(np.zeros(16))
        
        max_length = max(len(f) for f in features)
        padded_features = []
        for feature in features:
            padded = np.pad(feature, (0, max_length - len(feature)), 'constant')
            padded_features.append(padded)
        
        return np.array(padded_features)
    
    def build_graph_structure(self, signals, labels, save_path=None):
        print("Building graph structure...")
        
        features = self.extract_features_from_signals(signals)
        print(f"Extracted features shape: {features.shape}")
        
        print("Learning intelligent graph with DER...")
        adjacency_matrix = self.der.learn_intelligent_graph(features)
        print(f"Adjacency matrix shape: {adjacency_matrix.shape}")
        
        edge_index = self.adjacency_to_edge_index(adjacency_matrix)
        
        x = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)
        
        graph_data = Data(x=x, edge_index=edge_index, y=y)
        
        if save_path:
            self.save_graph_data(graph_data, adjacency_matrix, save_path)
        
        return graph_data, adjacency_matrix
    
    def adjacency_to_edge_index(self, adjacency_matrix):
        threshold = np.percentile(adjacency_matrix[adjacency_matrix > 0], 80)
        adjacency_matrix[adjacency_matrix < threshold] = 0
        
        edge_indices = np.nonzero(adjacency_matrix)
        edge_index = torch.tensor(np.vstack(edge_indices), dtype=torch.long)
        return edge_index
    
    def save_graph_data(self, graph_data, adjacency_matrix, save_path):
        data_dict = {
            'graph_data': graph_data,
            'adjacency_matrix': adjacency_matrix,
            'num_nodes': graph_data.x.shape[0],
            'num_features': graph_data.x.shape[1],
            'num_classes': len(torch.unique(graph_data.y))
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(data_dict, f)
        print(f"Graph data saved to {save_path}")
    
    def load_graph_data(self, load_path):
        with open(load_path, 'rb') as f:
            data_dict = pickle.load(f)
        return data_dict
    
    def optimize_and_save_graphs(self):
        print("Loading data...")
        train_signals, train_labels = self.data_loader.load_train_data()
        test_signals, test_labels = self.data_loader.load_test_data()
        
        print(f"Train data shape: {train_signals.shape}, Train labels shape: {train_labels.shape}")
        print(f"Test data shape: {test_signals.shape}, Test labels shape: {test_labels.shape}")
        
        train_signals = self.data_loader.normalize_signals(train_signals)
        test_signals = self.data_loader.normalize_signals(test_signals)
        
        print("Building training graph...")
        train_graph, train_adj = self.build_graph_structure(
            train_signals, train_labels, 'train_graph.pkl'
        )
        
        print("Building test graph...")
        test_graph, test_adj = self.build_graph_structure(
            test_signals, test_labels, 'test_graph.pkl'
        )
        
        print("Graph optimization completed!")
        print(f"Train graph: {train_graph.x.shape[0]} nodes, {train_graph.edge_index.shape[1]} edges")
        print(f"Test graph: {test_graph.x.shape[0]} nodes, {test_graph.edge_index.shape[1]} edges")
        
        return train_graph, test_graph

if __name__ == "__main__":
    optimizer = GraphOptimizer()
    optimizer.optimize_and_save_graphs() 