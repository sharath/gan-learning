import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, model_dim, data_dim):
        super(Generator, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(latent_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, data_dim)
        )
    
    def forward(self, x):
        return self.seq(x)
    
    
class Discriminator(nn.Module):
    def __init__(self, model_dim, data_dim):
        super(Discriminator, self).__init__()
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
    
    def forward(self, x):
        return self.seq(x)
    
    
class Classifier(nn.Module):
    def __init__(self, model_dim, data_dim, num_classes):
        super(Classifier, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(data_dim, model_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(model_dim, model_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(model_dim, model_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(model_dim, num_classes),
            nn.Softmax(1)
        )
    
    def forward(self, x):
        return self.seq(x)