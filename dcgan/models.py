import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as SN
from utils import xavier_init, gan_init

class Generator(nn.Module):
    def __init__(self, latent_dim, model_dim, data_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, model_dim * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(model_dim * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(model_dim * 8, model_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(model_dim * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(model_dim * 4, model_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(model_dim * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(model_dim * 2, model_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(model_dim),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(model_dim, data_dim, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self.apply(gan_init)

    def forward(self, x):
        x = x.view(-1, self.latent_dim, 1, 1)
        x = self.main(x)
        return x.view(x.shape[0], -1)
    
class Discriminator(nn.Module):
    def __init__(self, model_dim, data_dim, conditional=False, spec_norm=False):
        super(Discriminator, self).__init__()
        self.data_dim = data_dim
        if spec_norm:
            self.main = nn.Sequential(
                SN(nn.Conv2d(data_dim+1 if conditional else data_dim, model_dim, 4, 2, 1, bias=False)),
                nn.LayerNorm(32, 32),
                nn.LeakyReLU(0.2, inplace=True),
        
                SN(nn.Conv2d(model_dim, model_dim * 2, 4, 2, 1, bias=False)),
                nn.LayerNorm(16, 16),
                nn.LeakyReLU(0.2, inplace=True),
        
                SN(nn.Conv2d(model_dim * 2,model_dim * 4, 4, 2, 1, bias=False)),
                nn.LayerNorm(8, 8),
                nn.LeakyReLU(0.2, inplace=True),
        
                SN(nn.Conv2d(model_dim * 4, model_dim * 8, 4, 2, 1, bias=False)),
                nn.LayerNorm(4, 4),
                nn.LeakyReLU(0.2, inplace=True),
         
                SN(nn.Conv2d(model_dim * 8, 1, 4, 1, 0, bias=False)),
                nn.Sigmoid(),
                nn.Flatten()
            )
        
            self.cond_upscale = SN(nn.Linear(10, 64*64)) if conditional else None
        else:
            self.main = nn.Sequential(
                nn.Conv2d(data_dim+1 if conditional else data_dim, model_dim, 4, 2, 1, bias=False),
                nn.LayerNorm(32, 32),
                nn.LeakyReLU(0.2, inplace=True),
        
                nn.Conv2d(model_dim, model_dim * 2, 4, 2, 1, bias=False),
                nn.LayerNorm(16, 16),
                nn.LeakyReLU(0.2, inplace=True),
        
                nn.Conv2d(model_dim * 2,model_dim * 4, 4, 2, 1, bias=False),
                nn.LayerNorm(8, 8),
                nn.LeakyReLU(0.2, inplace=True),
        
                nn.Conv2d(model_dim * 4, model_dim * 8, 4, 2, 1, bias=False),
                nn.LayerNorm(4, 4),
                nn.LeakyReLU(0.2, inplace=True),
         
                nn.Conv2d(model_dim * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid(),
                nn.Flatten()
            )
            self.cond_upscale = nn.Linear(10, 64*64) if conditional else None
        self.apply(gan_init)
            
    def forward(self, x, c=None):
        x = x.view(-1, self.data_dim, 64, 64)
        if c is not None:
            c = self.cond_upscale(c).view(-1, 1, 64, 64)
            return self.main(torch.cat([x, c], dim=1))
        return self.main(x)
    
class Classifier(nn.Module):
    def __init__(self, model_dim, data_dim, num_classes):
        super(Classifier, self).__init__()
        self.data_dim = data_dim
        self.seq = nn.Sequential(
            nn.Conv2d(data_dim, 6, 5),
            nn.BatchNorm2d(6),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(6, 16, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(16, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            
            nn.Flatten(),
            
            nn.Linear(32 * 4 * 4, 120),
            nn.BatchNorm1d(120),
            nn.ReLU(True),
            
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(True),
            
            nn.Linear(84, num_classes)
        )
        
        self.apply(xavier_init)

    def forward(self, x):       
        x = x.view(-1, self.data_dim, 64, 64)
        return self.seq(x)
