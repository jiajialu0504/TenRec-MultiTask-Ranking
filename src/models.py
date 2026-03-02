import torch
import torch.nn as nn

class CrossNetV2(nn.Module):
    """WWW 2021: DCN-v2 显式特征交叉层"""
    def __init__(self, input_dim):
        super(CrossNetV2, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.bias = nn.Parameter(torch.Tensor(input_dim))
        nn.init.xavier_normal_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        x_w = torch.matmul(x, self.weight) + self.bias
        return x * x_w + x 

class MultiTaskPLEModel(nn.Module):
    """缝合模型: DCN-v2 + PLE (Progressive Layered Extraction)"""
    def __init__(self, sparse_config, sparse_features, embed_dim=16):
        super(MultiTaskPLEModel, self).__init__()
        self.sparse_features = sparse_features
        self.embed_layers = nn.ModuleDict({
            feat: nn.Embedding(num_embeddings=val, embedding_dim=embed_dim)
            for feat, val in sparse_config.items()
        })
        
        self.input_dim = len(sparse_config) * embed_dim + 1 
        self.cross_net = CrossNetV2(self.input_dim)
        
        expert_dim = 64
        # PLE 专家系统: 1共享 + 4专用
        self.ex_shared = nn.Linear(self.input_dim, expert_dim)
        self.ex_click = nn.Linear(self.input_dim, expert_dim)
        self.ex_follow = nn.Linear(self.input_dim, expert_dim)
        self.ex_like = nn.Linear(self.input_dim, expert_dim)
        self.ex_share = nn.Linear(self.input_dim, expert_dim)

        # 门控机制
        self.gate_click = nn.Sequential(nn.Linear(self.input_dim, 2), nn.Softmax(dim=-1))
        self.gate_follow = nn.Sequential(nn.Linear(self.input_dim, 2), nn.Softmax(dim=-1))
        self.gate_like = nn.Sequential(nn.Linear(self.input_dim, 2), nn.Softmax(dim=-1))
        self.gate_share = nn.Sequential(nn.Linear(self.input_dim, 2), nn.Softmax(dim=-1))

        def build_tower():
            return nn.Sequential(nn.Linear(expert_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
        
        self.tower_click = build_tower()
        self.tower_follow = build_tower()
        self.tower_like = build_tower()
        self.tower_share = build_tower()

    def forward(self, x_sparse, x_dense):
        embeds = [self.embed_layers[feat](x_sparse[:, i]) for i, feat in enumerate(self.sparse_features)]
        x = torch.cat(embeds + [x_dense], dim=-1)
        x_crossed = self.cross_net(x)
        
        s_out = torch.relu(self.ex_shared(x_crossed))
        def gate_combine(gate, specific_expert):
            w = gate(x_crossed)
            return w[:, 0:1] * s_out + w[:, 1:2] * torch.relu(specific_expert(x_crossed))

        return {
            'click': self.tower_click(gate_combine(self.gate_click, self.ex_click)),
            'follow': self.tower_follow(gate_combine(self.gate_follow, self.ex_follow)),
            'like': self.tower_like(gate_combine(self.ex_like, self.ex_like)), # 修正为对应专家
            'share': self.tower_share(gate_combine(self.gate_share, self.ex_share))
        }

class SharedBottomModel(nn.Module):
    """对比模型: 工业界最基础的 Shared-Bottom 架构"""
    def __init__(self, sparse_config, sparse_features, embed_dim=16):
        super(SharedBottomModel, self).__init__()
        self.sparse_features = sparse_features
        self.embed_layers = nn.ModuleDict({
            feat: nn.Embedding(num_embeddings=val, embedding_dim=embed_dim)
            for feat, val in sparse_config.items()
        })
        self.input_dim = len(sparse_config) * embed_dim + 1 
        self.bottom = nn.Sequential(nn.Linear(self.input_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())

        def build_tower():
            return nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
        
        self.tower_click, self.tower_follow = build_tower(), build_tower()
        self.tower_like, self.tower_share = build_tower(), build_tower()

    def forward(self, x_sparse, x_dense):
        embeds = [self.embed_layers[feat](x_sparse[:, i]) for i, feat in enumerate(self.sparse_features)]
        x = torch.cat(embeds + [x_dense], dim=-1)
        shared_out = self.bottom(x)
        return {
            'click': self.tower_click(shared_out),
            'follow': self.tower_follow(shared_out),
            'like': self.tower_like(shared_out),
            'share': self.tower_share(shared_out)
        }