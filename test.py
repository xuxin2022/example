import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import numpy as np

# 定义与训练时相同的模型结构
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2)

    def forward(self, x):

        x = self.fc1(x)
        return x

# 加载数据（只需要特征）
new_data_df = pd.read_csv('prediction.csv')  # 确保你的新数据文件名是正确的
X_new = new_data_df.values

# 数据标准化
scaler = StandardScaler()
X_new = scaler.fit_transform(X_new)

# 转换为Tensor
X_new_tensor = torch.tensor(X_new, dtype=torch.float32)

# 加载训练好的模型
input_dim = X_new.shape[1]
model = SimpleNN(input_dim)
model.load_state_dict(torch.load('best_model_1.pth'))
model.eval()

# 使用GPU进行预测
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model.to(device)
X_new_tensor = X_new_tensor.to(device)

# 进行预测
with torch.no_grad():
    outputs = model(X_new_tensor)
    _, predicted = torch.max(outputs, 1)

# 将预测结果转换为0和1
predictions = predicted.cpu().numpy()
print(predictions)
