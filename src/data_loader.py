import torch
from torch.utils.data import Dataset

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