import torch
import torch.nn as nn

class T2V(nn.Module):
    def __init__(self, n_feat):
        super(T2V, self).__init__()
        
        prob_dropout = 0.1

        self.t2v_ = nn.Sequential(
            nn.Linear(n_feat, int(n_feat/2)), 
            nn.BatchNorm1d(int(n_feat/2)),
            nn.ReLU(True),
            nn.Linear(int(n_feat/2), int(n_feat/4)),
            nn.BatchNorm1d(int(n_feat/4)),
            nn.Dropout(prob_dropout),
            nn.ReLU(True),
            nn.Linear(int(n_feat/4), int(n_feat/2)),
            nn.BatchNorm1d(int(n_feat/2)),
            nn.Dropout(prob_dropout),
            nn.ReLU(True),
            nn.Linear(int(n_feat/2), int(n_feat)),
            nn.BatchNorm1d(int(n_feat)),
            nn.Dropout(prob_dropout),
            nn.ReLU(True),
            nn.Linear(int(n_feat), int(n_feat*2)),
            nn.BatchNorm1d(int(n_feat*2)),
            nn.ReLU(True),
            nn.Linear(int(n_feat*2), int(n_feat*4))
         )
        
    def forward(self, x):
        x = torch.tensor(x).float().cuda()
        x = self.t2v_(x)
        return x
