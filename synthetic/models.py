import torch
import torch.nn as nn
from utils import gan_init, xavier_init
from torch.nn.utils import spectral_norm as SN


class Generator(nn.Module):
    def __init__(self, latent_dim, model_dim, data_dim):
        super(Generator, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(latent_dim, model_dim),
            nn.BatchNorm1d(model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim),
            nn.BatchNorm1d(model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim),
            nn.BatchNorm1d(model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, data_dim)
        ).apply(gan_init)
    
    def forward(self, x):
        return self.seq(x)
    
    
class Discriminator(nn.Module):
    def __init__(self, model_dim, data_dim, spec_norm=False):
        super(Discriminator, self).__init__()
        
        if spec_norm:
            self.seq = nn.Sequential(
                SN(nn.Linear(data_dim, model_dim)),
                nn.LayerNorm(model_dim),
                nn.LeakyReLU(0.2),
                SN(nn.Linear(model_dim, model_dim)),
                nn.LayerNorm(model_dim),
                nn.LeakyReLU(0.2),
                SN(nn.Linear(model_dim, model_dim)),
                nn.LayerNorm(model_dim),
                nn.LeakyReLU(0.2),
                SN(nn.Linear(model_dim, 1)),
                nn.Sigmoid()
            ).apply(gan_init)
        
        else:
            self.seq = nn.Sequential(
                nn.Linear(data_dim, model_dim),
                nn.LayerNorm(model_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(model_dim, model_dim),
                nn.LayerNorm(model_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(model_dim, model_dim),
                nn.LayerNorm(model_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(model_dim, 1),
                nn.Sigmoid()
            ).apply(gan_init)
        
    def forward(self, x):
        return self.seq(x)
    
    
    
    
class Classifier(nn.Module):
    def __init__(self, model_dim, data_dim, num_classes):
        super(Classifier, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(data_dim, model_dim),
            nn.BatchNorm1d(model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim),
            nn.BatchNorm1d(model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim),
            nn.BatchNorm1d(model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, num_classes),
            nn.Softmax(1)
        ).apply(xavier_init)
    
    def forward(self, x):
        return self.seq(x)