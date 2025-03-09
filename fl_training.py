import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from utils import load_and_preprocess_data
from models.fl_mlp import MLP, train, evaluate, FederatedClient




# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据
file_path = "data/fraud_tx_data.csv"
df = load_and_preprocess_data(file_path)

# 数据集拆分
X = df.drop(columns=["isFraud"]).values  # 特征
y = df["isFraud"].values  # 目标标签
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 转换为PyTorch张量
tensor_x_train = torch.tensor(X_train, dtype=torch.float32)
tensor_y_train = torch.tensor(y_train, dtype=torch.long)
tensor_x_test = torch.tensor(X_test, dtype=torch.float32)
tensor_y_test = torch.tensor(y_test, dtype=torch.long)

# 构建数据集与DataLoader
dataset_train = TensorDataset(tensor_x_train, tensor_y_train)
dataset_test = TensorDataset(tensor_x_test, tensor_y_test)
dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=True)

# ========== 训练集中式模型 ========== #
print("\nTraining Centralized Model...")
central_model = MLP(input_dim=X_train.shape[1], hidden_dim=64, output_dim=2)
central_loss, central_f1 = train(central_model, dataloader_train, epochs=50, lr=0.001, device=device)
central_f1_score = evaluate(central_model, dataloader_test, is_cl=True , device=device)



# 创建联邦客户端
num_clients = 5
client_dataloaders = torch.utils.data.random_split(dataset_train, [len(dataset_train)//num_clients]*num_clients)
clients = [FederatedClient(MLP(input_dim=X_train.shape[1], hidden_dim=64, output_dim=2), DataLoader(data, batch_size=32, shuffle=True)) for data in client_dataloaders]

# 联邦学习训练
print("\nTraining Federated Model...")
global_model = MLP(input_dim=X_train.shape[1], hidden_dim=64, output_dim=2)
federated_loss = []
federated_f1 = []

for round in range(10):  # 10 轮联邦训练
    local_weights = []
    for client in clients:
        client.local_update(epochs=10, lr=0.001, device=device)
        local_weights.append({key: value.clone() for key, value in client.model.state_dict().items()})
    
    # 服务器聚合权重（FedAvg）
    new_state_dict = {key: sum(w[key] for w in local_weights) / num_clients for key in local_weights[0]}
    global_model.load_state_dict(new_state_dict)
    
    loss, f1 = train(global_model, dataloader_train, epochs=10, lr=0.001, device=device)
    federated_loss.append(loss[-1])
    federated_f1.append(f1[-1])
    print(f"Round {round+1} completed")

federated_f1_score = evaluate(global_model, dataloader_test, is_cl=False, device=device)

# 绘制loss曲线与F1-score曲线
plt.figure()
fig, ax1 = plt.subplots()
ax2 = ax1.twiny()

ax1.plot(range(1, len(central_loss) + 1), central_loss, label="Centralized Loss", color='blue')
ax2.plot(range(1, len(federated_loss) + 1), federated_loss, label="Federated Loss", color='orange')
ax1.set_xlabel("Centralized Epoch")
ax2.set_xlabel("Federated Rounds")
plt.ylabel("Loss")
ax1.legend(loc='upper right', bbox_to_anchor=(1, 0.95))
ax2.legend(loc='upper right', bbox_to_anchor=(1, 0.88))
plt.title("Loss Curve Comparison with Dual X-axis")
plt.show()

plt.figure()
fig, ax1 = plt.subplots()
ax2 = ax1.twiny()

ax1.plot(range(1, len(central_f1) + 1), central_f1, label="Centralized F1-score", color='blue')
ax2.plot(range(1, len(federated_f1) + 1), federated_f1, label="Federated F1-score", color='orange')

ax1.set_xlabel("Centralized Epoch")
ax2.set_xlabel("Federated Rounds")
ax1.set_ylabel("F1-score")
ax1.legend(loc='lower right', bbox_to_anchor=(1, 0.12))
ax2.legend(loc='lower right', bbox_to_anchor=(1, 0.05))
plt.title("F1-score Curve Comparison with Dual X-axis")
plt.show()

