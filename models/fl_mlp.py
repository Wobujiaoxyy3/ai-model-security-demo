import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train(model, dataloader, epochs=50, lr=0.001, device="cpu"):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_history = []
    f1_history = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            predicted = torch.argmax(outputs, dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        f1 = f1_score(all_labels, all_preds)
        f1_history.append(f1)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, F1-score: {f1:.4f}")
    
    return loss_history, f1_history

from sklearn.metrics import f1_score, classification_report

def evaluate(model, dataloader, is_cl=True, device="cpu"):
    model = model.to(device)
    model.eval()
    
    correct, total = 0, 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            predicted = torch.argmax(outputs, dim=1)
            
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    # 计算指标
    accuracy = 100 * correct / total
    f1 = f1_score(all_labels, all_preds)  # 兼容多分类
    report = classification_report(all_labels, all_preds, digits=4)
    
    print(f"\n=== Model Evaluation ===")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"F1-score: {f1:.4f}")
    print(f"\nClassification Report:\n{report}")

    report_dict = classification_report(all_labels, all_preds, digits=4, output_dict=True)  
    report_df = pd.DataFrame(report_dict).transpose()

    if is_cl:
        report_df.to_csv("fl_analysis_results/cl_report.csv", index=True)
    else:
        report_df.to_csv("fl_analysis_results/fl_report.csv", index=True)

    return accuracy, f1, report


class FederatedClient:
    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader
    
    def local_update(self, epochs, lr, device="cpu"):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        for _ in range(epochs):
            for batch_x, batch_y in self.data_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()