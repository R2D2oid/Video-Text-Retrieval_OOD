'''
partially adopted from BarlowTwins at https://github.com/facebookresearch/barlowtwins
'''

import torch.nn as nn
import torch

class BarlowTwins(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        prob_dropout = 0.1

        n_feat = args.v_num_feats # 2048 -> 512
        self.v2t_1 = nn.Sequential(
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
        
        n_feat = args.t_num_feats # 512 -> 512
        self.t2v_2 = nn.Sequential(
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
            nn.Linear(int(n_feat/2), int(n_feat))
         )
                
        # projector
        sizes = [512] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, y1, y2):
        z1 = self.projector(self.v2t_1(y1))
        z2 = self.projector(self.t2v_2(y2))

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return loss
    
    def save(self, path_):
        torch.save(self.v2t_1.state_dict(), f'{path_}/model_v2t.sd')
        torch.save(self.t2v_2.state_dict(), f'{path_}/model_t2v.sd')
    
def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
