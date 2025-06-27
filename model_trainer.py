import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pickle
from drgan import DRGAN
from config import Config

class ModelTrainer:
    def __init__(self):
        self.config = Config()
        self.device = torch.device('cpu')  
        
    def load_graph_data(self, train_path, test_path):
        print("Loading preprocessed graph data...")
        
        with open(train_path, 'rb') as f:
            train_dict = pickle.load(f)
        
        with open(test_path, 'rb') as f:
            test_dict = pickle.load(f)
        
        train_graph = train_dict['graph_data']
        test_graph = test_dict['graph_data']
        
        print(f"Loaded train graph: {train_graph.x.shape[0]} nodes, {train_graph.edge_index.shape[1]} edges")
        print(f"Loaded test graph: {test_graph.x.shape[0]} nodes, {test_graph.edge_index.shape[1]} edges")
        
        return train_graph, test_graph, train_dict['num_classes']
    
    def create_model(self, input_dim, num_classes):
        model = DRGAN(input_dim, num_classes).to(self.device)
        return model
    
    def train_epoch(self, model, train_data, criterion, optimizer):
        model.train()
        train_data = train_data.to(self.device)
        
        optimizer.zero_grad()
        
        output = model(train_data.x, train_data.edge_index)
        loss = criterion(output, train_data.y)
        
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def evaluate_model(self, model, test_data):
        model.eval()
        test_data = test_data.to(self.device)
        
        with torch.no_grad():
            output = model(test_data.x, test_data.edge_index)
            predictions = torch.argmax(output, dim=1)
        
        y_true = test_data.y.cpu().numpy()
        y_pred = predictions.cpu().numpy()
        
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        
        return accuracy, report, cm
    
    def train_model(self, train_graph, test_graph, num_classes):
        input_dim = train_graph.x.shape[1]
        
        print(f"Creating model with input_dim={input_dim}, num_classes={num_classes}")
        model = self.create_model(input_dim, num_classes)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
        print(f"Using device: {self.device}")
        
        best_accuracy = 0
        
        for epoch in range(self.config.EPOCHS):
            train_loss = self.train_epoch(model, train_graph, criterion, optimizer)
            
            if (epoch + 1) % 10 == 0:
                accuracy, report, cm = self.evaluate_model(model, test_graph)
                print(f'Epoch {epoch+1}/{self.config.EPOCHS}:')
                print(f'  Train Loss: {train_loss:.4f}')
                print(f'  Test Accuracy: {accuracy:.4f}')
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                
                print('-' * 50)
        
        print(f'Training completed! Best Test Accuracy: {best_accuracy:.4f}')
        
        # 最终评估
        final_accuracy, final_report, final_cm = self.evaluate_model(model, test_graph)
        
        print("\nFinal Evaluation Results:")
        print(f"Final Test Accuracy: {final_accuracy:.4f}")
        print("\nClassification Report:")
        print(final_report)
        print("\nConfusion Matrix:")
        print(final_cm)
        
        return model, best_accuracy
    
    def run_training(self, train_graph_path='train_graph.pkl', test_graph_path='test_graph.pkl'):
        try:
            train_graph, test_graph, num_classes = self.load_graph_data(
                train_graph_path, test_graph_path
            )
            
            model, best_accuracy = self.train_model(train_graph, test_graph, num_classes)
            
            return model, best_accuracy
            
        except FileNotFoundError:
            print("Graph data files not found. Please run graph_optimizer.py first.")
            return None, 0
        except Exception as e:
            print(f"Training failed with error: {e}")
            import traceback
            traceback.print_exc()
            return None, 0

if __name__ == "__main__":
    trainer = ModelTrainer()
    model, accuracy = trainer.run_training() 