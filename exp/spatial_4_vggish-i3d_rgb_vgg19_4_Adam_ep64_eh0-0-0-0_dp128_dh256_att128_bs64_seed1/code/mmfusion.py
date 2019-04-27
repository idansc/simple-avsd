import torch
import torch.nn as nn
import torch.nn.functional as F

class MMFusion(nn.Module):
    def __init__(self, concat_dim, output_dim):
        super(MMFusion, self).__init__()
        self.concat_dim = concat_dim
        self.output_dim = output_dim
        self.fusion_model = nn.Sequential(
            nn.Linear(concat_dim, (concat_dim) // 2, bias=False),
            nn.BatchNorm1d((concat_dim) // 2),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear((concat_dim) // 2, output_dim)
        )


    def forward(self,  feat):
        return self.fusion_model(feat)
