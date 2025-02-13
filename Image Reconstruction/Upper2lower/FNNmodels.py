import torch
import torch.nn as nn

# option LeakyReLU --> ReLU

class Generator(nn.Module):
    def __init__(self, m, r, q):
        super(Generator, self).__init__()

        self.m = m
        self.r = r
        self.q = q
        
        self.conditional_generate_sample = nn.Sequential(
            nn.Linear(m+r, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, q),
            nn.Tanh()
        )

        self._initialize_weights()

    def forward(self, Eta, Z): # z.size (bs, r)
        generative_result = self.conditional_generate_sample(torch.concat([Eta, Z], dim=-1))
        generative_result = generative_result.view(generative_result.size(0), 1, 14, 28)
        return generative_result
    
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
            nn.Linear(d+q, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            # nn.Sigmoid() # option
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
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, r)
        )

        self._initialize_weights()

    def forward(self, x): # x.size (bs, 1, 14, 28)
        x = x.view(x.size(0), -1)
        return self.tran_to_feature(x) # z.size (bs, r)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        