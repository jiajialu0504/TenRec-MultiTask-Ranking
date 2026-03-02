import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# 1. 数据加载与预处理
data = pd.read_csv('QK-video.csv', nrows=100000)

sparse_features = ['user_id', 'item_id', 'video_category', 'gender', 'age']
dense_features = ['watching_times']
target_features = ['click', 'follow', 'like', 'share']

# Sparse 编码
sparse_feature_config = {} 
for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])
    sparse_feature_config[feat] = data[feat].nunique()

# Dense 归一化
data['watching_times'] = np.log1p(data['watching_times'])
mms = MinMaxScaler(feature_range=(0, 1))
data[dense_features] = mms.fit_transform(data[dense_features])

# 数据切分
train_df, temp_df = train_test_split(data, test_size=0.2, random_state=2026)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=2026)


# 2. 类定义 (Dataset & Model)
class TenRecDataset(Dataset):
    def __init__(self, df, sparse_feats, dense_feats, target_feats):
        self.x_sparse = torch.LongTensor(df[sparse_feats].values)
        self.x_dense = torch.FloatTensor(df[dense_feats].values)
        self.y = torch.FloatTensor(df[target_feats].values)
        self.target_names = target_feats

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            'x_sparse': self.x_sparse[idx],
            'x_dense': self.x_dense[idx],
            'labels': {name: self.y[idx, i] for i, name in enumerate(self.target_names)}
        }

class SharedBottomModel(nn.Module):
    def __init__(self, sparse_config, embed_dim=16):
        super(SharedBottomModel, self).__init__()
        # 依然保留同样的 Embedding 层
        self.embed_layers = nn.ModuleDict({
            feat: nn.Embedding(num_embeddings=val, embedding_dim=embed_dim)
            for feat, val in sparse_config.items()
        })
        self.input_dim = len(sparse_config) * embed_dim + 1 
        
        # 共享底层 (没有 DCN-v2 和 PLE 专家)
        self.bottom = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # 任务塔
        def build_tower():
            return nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
        
        self.tower_click = build_tower()
        self.tower_follow = build_tower()
        self.tower_like = build_tower()
        self.tower_share = build_tower()

    def forward(self, x_sparse, x_dense):
        embeds = [self.embed_layers[feat](x_sparse[:, i]) for i, feat in enumerate(sparse_features)]
        x = torch.cat(embeds + [x_dense], dim=-1)
        shared_out = self.bottom(x)
        
        return {
            'click': self.tower_click(shared_out),
            'follow': self.tower_follow(shared_out),
            'like': self.tower_like(shared_out),
            'share': self.tower_share(shared_out)
        }

# ==========================================
# 3. 评估函数
# ==========================================
def evaluate_model(model, loader, device):
    model.eval()
    all_labels = {target: [] for target in target_features}
    all_preds = {target: [] for target in target_features}
    
    with torch.no_grad():
        for batch in loader:
            x_s, x_d = batch['x_sparse'].to(device), batch['x_dense'].to(device)
            labels = batch['labels']
            outputs = model(x_s, x_d)
            
            for target in target_features:
                all_labels[target].extend(labels[target].numpy())
                all_preds[target].extend(outputs[target].cpu().squeeze().numpy())
    
    auc_results = {}
    for target in target_features:
        try:
            auc_results[target] = roc_auc_score(all_labels[target], all_preds[target])
        except ValueError:
            auc_results[target] = 0.5 
            
    return auc_results

# ==========================================
# 4. 实例化与训练准备
# ==========================================
train_loader = DataLoader(TenRecDataset(train_df, sparse_features, dense_features, target_features), batch_size=1024, shuffle=True)
val_loader = DataLoader(TenRecDataset(val_df, sparse_features, dense_features, target_features), batch_size=1024)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SharedBottomModel(sparse_feature_config).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()

print(f"Baseline 模型初始化完成，运行设备: {device}")

# ==========================================
# 5. 训练循环
# ==========================================
epochs = 5
print("开始训练 Baseline 实验组...")

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        x_s, x_d = batch['x_sparse'].to(device), batch['x_dense'].to(device)
        labels = {k: v.to(device).unsqueeze(1) for k, v in batch['labels'].items()}
        
        optimizer.zero_grad()
        outputs = model(x_s, x_d)
        
        # --- 修改点：Baseline 直接求和，不使用 log_vars ---
        total_loss = 0
        for target in target_features:
            task_loss = loss_fn(outputs[target], labels[target])
            total_loss += task_loss 
        
        total_loss.backward()
        optimizer.step()
        epoch_loss += total_loss.item()
    
    # 每个 Epoch 评估一次
    val_aucs = evaluate_model(model, val_loader, device)
    
    print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(train_loader):.4f}")
    for target, score in val_aucs.items():
        print(f"  {target} AUC: {score:.4f}")
    print("-" * 30)

print("Baseline 实验组训练完成！请记录以上 AUC 数据。")