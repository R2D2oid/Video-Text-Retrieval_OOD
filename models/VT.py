'''
partially adopted from BarlowTwins at https://github.com/facebookresearch/barlowtwins
'''

import torch.nn as nn
import torch

from models.contrastive_loss import ContrastiveLoss

class VT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        prob_dropout = 0.1

        n_feat = args.v_num_feats # 2048 -> 512
        self.v2r = nn.Sequential(
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
        self.t2r = nn.Sequential(
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
        
        if args.loss_criterion == 'cross_correlation':
            # the loss is calculated in forward atm 
            self.criterion = None
        elif args.loss_criterion == 'contrastive':
            self.criterion = ContrastiveLoss(temperature=0.5, contrast_mode='all', base_temperature=1)
        elif args.loss_criterion == 'cosine':
            self.criterion = nn.CosineEmbeddingLoss()
            # the cosine embedding loss takes a target y=1 for training positive (similar) vectors 
            #                                      and y=-1 for training dissimilar (negative) vectors
            self.target_tensor = torch.Tensor(1)
        else: # mse
            self.criterion = nn.MSELoss()
        

    def forward(self, y1, y2):
        z1 = self.projector(self.v2r(y1))
        z2 = self.projector(self.t2r(y2))

        proj_v2r = self.bn(z1)
        proj_t2r = self.bn(z2)
        
        if self.args.loss_criterion == 'cross_correlation':
            # empirical cross-correlation matrix
            c = proj_v2r.T @ proj_t2r
            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = off_diagonal(c).pow_(2).sum()
            loss = on_diag + self.args.lambd * off_diag
        elif self.args.loss_criterion == 'contrastive':
            p_v = proj_v2r.unsqueeze(dim=1)
            p_t = proj_t2r.unsqueeze(dim=1)
            loss = self.criterion(torch.cat([p_v,p_t], dim=1))
        elif self.args.loss_criterion == 'cosine':
            loss = self.criterion(proj_v2r, proj_t2r, target_tensor)
        else: # mse
            loss = self.criterion(proj_v2r, proj_t2r)
        
        return loss
    
    def save(self, path_):
        torch.save(self.v2r.state_dict(), f'{path_}/model_v2r.sd')
        torch.save(self.t2r.state_dict(), f'{path_}/model_t2r.sd')
        torch.save(self.state_dict(), f'{path_}/model_vt.sd')

    def get_v_and_t_representation(self, y1, y2):
        # proj_v2r = self.bn(self.projector(self.v2r(y1)))
        # proj_t2r = self.bn(self.projector(self.t2r(y2)))
        proj_v2r = self.v2r(y1)
        proj_t2r = self.t2r(y2)
        
        return proj_v2r, proj_t2r
    
    def get_weights(self):
        all_weights = []
        for l in self.v2r:
            if hasattr(l, 'weight'):
                all_weights.append(l.weight)
        for l in self.t2r:
            if hasattr(l, 'weight'):
                all_weights.append(l.weight)
        for l in self.projector:
            if hasattr(l, 'weight'):
                all_weights.append(l.weight)
        return all_weights

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()