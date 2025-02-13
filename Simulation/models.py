import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# q: response_dim
# d: covariate_dim
# m: latent_dim
# r: intrinsic_dim
class Generator(nn.Module):
    def __init__(self, m, r, q):
        super(Generator, self).__init__()

        self.m = m
        self.r = r
        self.q = q
        
        self.conditional_generate_sample = nn.Sequential(
            nn.Linear(m+r, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, q)
        )
        
        self._initialize_weights()

    def forward(self, Eta, Z):
        # Return generated response
        return self.conditional_generate_sample(torch.concat([Eta, Z], dim=-1))
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class Discriminator(nn.Module):
    def __init__(self, d, q):
        super(Discriminator, self).__init__()

        self.d = d
        self.q = q

        self.feature_to_prob = nn.Sequential(
            nn.Linear(d+q, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self._initialize_weights()

    def forward(self, x):
        return self.feature_to_prob(x)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
class Representation(nn.Module):
    def __init__(self, d, r):
        super(Representation, self).__init__()

        self.d = d
        self.r = r

        self.tran_to_feature = nn.Sequential(
            nn.Linear(d, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, r)
        )
        
        self._initialize_weights()

    def forward(self, x):
        return self.tran_to_feature(x)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        
