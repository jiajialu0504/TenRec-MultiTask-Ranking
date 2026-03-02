import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

# 允许导入 src 目录下的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models import MultiTaskPLEModel
from src.data_loader import TenRecDataset
from src.utils import evaluate_model

# 配置
FILE_PATH = 'QK-video.csv'
SPARSE_FEATURES = ['user_id', 'item_id', 'video_category', 'gender', 'age']
DENSE_FEATURES = ['watching_times']
TARGET_FEATURES = ['click', 'follow', 'like', 'share']

# 1. 数据预处理
data = pd.read_csv(FILE_PATH, nrows=100000)
sparse_config = {} 
for feat in SPARSE_FEATURES:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])
    sparse_config[feat] = data[feat].nunique()

data['watching_times'] = np.log1p(data['watching_times'])
mms = MinMaxScaler(feature_range=(0, 1))
data[DENSE_FEATURES] = mms.fit_transform(data[DENSE_FEATURES])

train_df, temp_df = train_test_split(data, test_size=0.2, random_state=2026)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=2026)

# 2. 准备数据加载器
train_loader = DataLoader(TenRecDataset(train_df, SPARSE_FEATURES, DENSE_FEATURES, TARGET_FEATURES), batch_size=1024, shuffle=True)
val_loader = DataLoader(TenRecDataset(val_df, SPARSE_FEATURES, DENSE_FEATURES, TARGET_FEATURES), batch_size=1024)

# 3. 实例化模型与优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiTaskPLEModel(sparse_config, SPARSE_FEATURES).to(device)

# 缝合点：UWL 动态权重参数
log_vars = nn.Parameter(torch.zeros(len(TARGET_FEATURES), device=device))
optimizer = torch.optim.Adam(list(model.parameters()) + [log_vars], lr=0.001)
loss_fn = nn.BCELoss()

# 4. 训练循环
epochs = 5
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        x_s, x_d = batch['x_sparse'].to(device), batch['x_dense'].to(device)
        labels = {k: v.to(device).unsqueeze(1) for k, v in batch['labels'].items()}
        
        optimizer.zero_grad()
        outputs = model(x_s, x_d)
        
        total_loss = 0
        for i, target in enumerate(TARGET_FEATURES):
            task_loss = loss_fn(outputs[target], labels[target])
            # UWL 动态权重公式
            weighted_loss = torch.exp(-log_vars[i]) * task_loss + log_vars[i]
            total_loss += weighted_loss
        
        total_loss.backward()
        optimizer.step()
    
    val_aucs = evaluate_model(model, val_loader, device, TARGET_FEATURES)
    print(f"Epoch {epoch+1} | Click AUC: {val_aucs['click']:.4f} | Share AUC: {val_aucs['share']:.4f}")

print("Full Model 训练完成！")