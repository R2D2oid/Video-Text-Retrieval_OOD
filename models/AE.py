import torch
import torch.nn as nn

class AE(nn.Module):
    def __init__(self, n_feat):
        super(AE, self).__init__()
        
        # the layer dim reductions are based on 2048 video feat vector size and 512 text feature vector size
        
        self.encoder_ = nn.Sequential(
            nn.Linear(n_feat, int(n_feat/2)),
            nn.Dropout(0.2),
            nn.BatchNorm1d(int(n_feat/2)),
            nn.ReLU(True),
            nn.Linear(int(n_feat/2), int(n_feat/2)),
            nn.ReLU(True),
            nn.Linear(int(n_feat/2), int(n_feat/4)),
            nn.Dropout(0.1),
            nn.ReLU(True),
            nn.Linear(int(n_feat/4), int(n_feat/8)),
            nn.ReLU(True),
            nn.Linear(int(n_feat/8), 32)
         )
        
        self.decoder_ = nn.Sequential(
            nn.Linear(32, int(n_feat/8)),
            nn.ReLU(True),
            nn.Linear(int(n_feat/8), int(n_feat/4)),
            nn.Dropout(0.1),
            nn.ReLU(True),
            nn.Linear(int(n_feat/4), int(n_feat/2)),
            nn.ReLU(True),
            nn.Linear(int(n_feat/2), int(n_feat/2)),
            nn.ReLU(True),
            nn.BatchNorm1d(int(n_feat/2)),
            nn.Dropout(0.2),
            nn.Linear(int(n_feat/2), n_feat)
        )
         
    def encoder(self, x):
        x = torch.tensor(x).float().cuda()
        x = self.encoder_(x)
        # add L2 normalization to get L2_norm(embedding) = 1
        return x
    
    def decoder(self, x):
        x = torch.tensor(x).float().cuda()
        x = self.decoder_(x)
        return x
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
