import torch.nn as nn
from utils import weights_init

class Generator(nn.Module):
    def __init__(self, latent_dim, model_dim, data_dim, spec_norm=False):
        super(Generator, self).__init__()
        if spec_norm is False:
            self.seq = nn.Sequential(
                nn.Linear(latent_dim, model_dim),
                nn.ReLU(),
                nn.Linear(model_dim, model_dim),
                nn.ReLU(),
                nn.Linear(model_dim, model_dim),
                nn.ReLU(),
                nn.Linear(model_dim, data_dim)
            )
        else:
            self.seq = nn.Sequential(
                nn.utils.spectral_norm(nn.Linear(latent_dim, model_dim)),
                nn.ReLU(),
                nn.utils.spectral_norm(nn.Linear(model_dim, model_dim)),
                nn.ReLU(),
                nn.utils.spectral_norm(nn.Linear(model_dim, model_dim)),
                nn.ReLU(),
                nn.utils.spectral_norm(nn.Linear(model_dim, data_dim))
            )#.apply(weights_init)
    
    def forward(self, x):
        return self.seq(x)
    
class Discriminator(nn.Module):
    def __init__(self, model_dim, data_dim, spec_norm=False):
        super(Discriminator, self).__init__()
        if spec_norm is False:
            self.seq = nn.Sequential(
                nn.Linear(data_dim, model_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(model_dim, model_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(model_dim, model_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(model_dim, 1),
                nn.Sigmoid()
            )
            
        else:
            self.seq = nn.Sequential(
                nn.utils.spectral_norm(nn.Linear(data_dim, model_dim)),
                nn.LeakyReLU(0.2),
                nn.utils.spectral_norm(nn.Linear(model_dim, model_dim)),
                nn.LeakyReLU(0.2),
                nn.utils.spectral_norm(nn.Linear(model_dim, model_dim)),
                nn.LeakyReLU(0.2),
                nn.utils.spectral_norm(nn.Linear(model_dim, 1)),
                #nn.Linear(model_dim, 1),
                nn.Sigmoid()
            )#.apply(weights_init)
            
    
    def forward(self, x):
        return self.seq(x)