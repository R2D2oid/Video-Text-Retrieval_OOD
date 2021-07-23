import torch
import torch.nn as nn

class V2T(nn.Module):
    def __init__(self, n_feat):
        super(V2T, self).__init__()
        
        prob_dropout = 0.1

        self.v2t_ = nn.Sequential(
            nn.Linear(n_feat, int(n_feat/2)),
            nn.BatchNorm1d(int(n_feat/2)),
            nn.ReLU(True),
            nn.Linear(int(n_feat/2), int(n_feat/4)),
            nn.BatchNorm1d(int(n_feat/4)),
            nn.Dropout(prob_dropout),
            nn.ReLU(True),
            nn.Linear(int(n_feat/4), int(n_feat/8)),
            nn.BatchNorm1d(int(n_feat/8)),
            nn.Dropout(prob_dropout),
            nn.ReLU(True),
            nn.Linear(int(n_feat/8), int(n_feat/16)),
            nn.BatchNorm1d(int(n_feat/16)),
            nn.Dropout(prob_dropout),
            nn.ReLU(True),
            nn.Linear(int(n_feat/16), int(n_feat/8)),
            nn.BatchNorm1d(int(n_feat/8)),
            nn.ReLU(True),
            nn.Linear(int(n_feat/8), int(n_feat/4)),
         )
        
    def forward(self, x):
        x = torch.tensor(x).float().cuda()
        x = self.v2t_(x)
        return x
