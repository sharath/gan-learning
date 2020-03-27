import torch
import torch.nn.functional as F

class UniformDatasetSampler:
    def __init__(self, dataset):
        self.dataset = dataset
        
    def get_batch(self, batch_size):
        idx = torch.randint(0, len(self.dataset), (batch_size, ))
        return self.dataset[idx].clone()
    
    
class UniformConditionalDatasetSampler:
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels
        self.num_classes = len(set(labels.tolist()))
        
    def get_batch(self, batch_size):
        idx = torch.randint(0, len(self.dataset), (batch_size, ))
        data = self.dataset[idx].clone()
        labels_oh = torch.nn.functional.one_hot(self.labels[idx].clone().long(), self.num_classes).float()
        return data, labels_oh
    
    
class UniformConditionalLatentSampler:
    def __init__(self, latent_dim, labels, low=-1, high=1):
        self.latent_dim = latent_dim
        self.low = low
        self.high = high
        self.num_classes = len(set(labels.tolist()))
        
    def get_batch(self, batch_size):
        rand_labels = torch.randint(0, self.num_classes, (batch_size, ))
        labels_oh = torch.nn.functional.one_hot(rand_labels, self.num_classes).float()
        rand_noise = (self.high - self.low) * torch.rand((batch_size, self.latent_dim)) + self.low
        return rand_noise, labels_oh
    

    
class UniformLatentSampler:
    def __init__(self, latent_dim, low=-1, high=1):
        self.latent_dim = latent_dim
        self.low = low
        self.high = high

    def get_batch(self, batch_size):
        return (self.high - self.low) * torch.rand((batch_size, self.latent_dim)) + self.low
    
    
class NormalLatentSampler:
    def __init__(self, latent_dim, mean=0, std=1):
        self.latent_dim = latent_dim
        self.mean = mean
        self.std = std

    def get_batch(self, batch_size):
        return self.std * torch.randn((batch_size, self.latent_dim)) + self.mean