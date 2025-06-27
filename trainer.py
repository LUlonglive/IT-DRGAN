import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from config import Config

class Trainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.config = Config()
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)
        
    def train_epoch(self, train_data):
        self.model.train()
        total_loss = 0
        
        train_data = train_data.to(self.device)
        
        self.optimizer.zero_grad()
        
        output = self.model(train_data.x, train_data.edge_index)
        loss = self.criterion(output, train_data.y)
        
        loss.backward()
        self.optimizer.step()
        
        total_loss += loss.item()
        
        return total_loss
    
    def evaluate(self, test_data):
        self.model.eval()
        
        test_data = test_data.to(self.device)
        
        with torch.no_grad():
            output = self.model(test_data.x, test_data.edge_index)
            predictions = torch.argmax(output, dim=1)
            
        y_true = test_data.y.cpu().numpy()
        y_pred = predictions.cpu().numpy()
        
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        
        return accuracy, report, cm
    
    def train(self, train_data, test_data):
        best_accuracy = 0
        
        for epoch in range(self.config.EPOCHS):
            train_loss = self.train_epoch(train_data)
            
            if (epoch + 1) % 10 == 0:
                accuracy, report, cm = self.evaluate(test_data)
                print(f'Epoch {epoch+1}/{self.config.EPOCHS}:')
                print(f'  Train Loss: {train_loss:.4f}')
                print(f'  Test Accuracy: {accuracy:.4f}')
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    torch.save(self.model.state_dict(), 'best_model.pth')
                
                print('-' * 50)
        
        print(f'Best Test Accuracy: {best_accuracy:.4f}')
        return best_accuracy 