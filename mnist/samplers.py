import torch
import numpy as np
import torch.nn.functional as F
  
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
    
    
class NormalConditionalLatentSampler:
    def __init__(self, latent_dim, labels, mean=0, std=1):
        self.latent_dim = latent_dim
        self.mean = mean
        self.std = std
        self.num_classes = len(set(labels.tolist()))
        
    def get_batch(self, batch_size):
        rand_labels = torch.randint(0, self.num_classes, (batch_size, ))
        labels_oh = torch.nn.functional.one_hot(rand_labels, self.num_classes).float()
        rand_noise = self.std * torch.randn((batch_size, self.latent_dim)) + self.mean
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
    
    
class ScanUniformDatasetSampler:
    def __init__(self, dataset):
        self.dataset = dataset
        self.order = np.random.permutation(len(self.dataset))
        
    def get_batch(self, batch_size):
        if batch_size > len(self.order):
            self.order = np.random.permutation(len(self.dataset))
        idx, self.order = np.split(self.order, [batch_size])
        return self.dataset[idx].clone()
    
    
class ScanUniformConditionalDatasetSampler:
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels
        self.num_classes = len(set(labels.tolist()))
        self.order = np.random.permutation(len(self.dataset))
        
    def get_batch(self, batch_size):
        if batch_size > len(self.order):
            self.order = np.random.permutation(len(self.dataset))
        idx, self.order = np.split(self.order, [batch_size])
        data = self.dataset[idx].clone()
        labels_oh = torch.nn.functional.one_hot(self.labels[idx].clone().long(), self.num_classes).float()
        return data, labels_oh
    
    
class DifficultyDatasetSampler:
    def __init__(self, dataset, eps=0):
        self.dataset = dataset
        
        self.indices = np.arange(len(self.dataset))
        self.probs = np.ones_like(self.indices) / len(self.indices)
        self.trace = np.ones_like(self.indices) / len(self.indices)
                
        self.last_idx = None
        self.eps = eps
        
    def get_batch(self, batch_size):
        if batch_size > np.sum(self.probs > 0):
            self.probs[self.probs == 0] = self.trace[self.probs == 0]
            self.trace = np.ones_like(self.indices) / len(self.indices)
            
        self.probs /= np.sum(self.probs)
        idx = np.random.choice(self.indices, size=(batch_size, ), replace=False, p=self.probs)
        self.probs[idx] = 0
        
        data = self.dataset[idx].clone()
        self.last_idx = idx
        
        return data
    
    def record(self, h):
        h = h.reshape(-1)
        self.trace[self.last_idx] = 1 - h + self.eps
        
    
class DifficultyConditionalDatasetSampler:
    def __init__(self, dataset, labels, eps=0):
        self.dataset = dataset
        self.labels = labels
        self.num_classes = len(set(labels.tolist()))
        
        self.indices = np.arange(len(self.dataset))
        self.probs = np.ones_like(self.indices) / len(self.indices)
        self.trace = np.ones_like(self.indices) / len(self.indices)
                
        self.last_idx = None
        self.eps = eps
        
    def get_batch(self, batch_size):
        if batch_size > np.sum(self.probs > 0):
            self.probs[self.probs == 0] = self.trace[self.probs == 0]
            self.trace = np.ones_like(self.indices) / len(self.indices)
            
        self.probs /= np.sum(self.probs)
        idx = np.random.choice(self.indices, size=(batch_size, ), replace=False, p=self.probs)
        self.probs[idx] = 0
        
        data = self.dataset[idx].clone()
        labels_oh = torch.nn.functional.one_hot(self.labels[idx].clone().long(), self.num_classes).float()
        
        self.last_idx = idx
        return data, labels_oh
    
    def record(self, h):
        h = h.reshape(-1)
        self.trace[self.last_idx] = 1 - h + self.eps
        
        
class DifficultyWeightedConditionalDatasetSampler:
    def __init__(self, dataset, labels, eps=0):
        self.dataset = dataset
        self.labels = labels
        self.num_classes = len(set(labels.tolist()))
        
        self.indices = np.arange(len(self.dataset))
        self.probs = np.ones_like(self.indices) / len(self.indices)
        self.weights = np.ones_like(self.indices) / len(self.indices)
                
        self.last_idx = None
        self.eps = eps
        
    def get_batch(self, batch_size):
        if batch_size > np.sum(self.probs > 0):
            self.probs = np.ones_like(self.indices) / len(self.indices)
            
        self.probs /= np.sum(self.probs)
        idx = np.random.choice(self.indices, size=(batch_size, ), replace=False, p=self.probs)
        self.probs[idx] = 0
        
        data = self.dataset[idx].clone()
        labels_oh = torch.nn.functional.one_hot(self.labels[idx].clone().long(), self.num_classes).float()
        
        self.last_idx = idx
        return data, labels_oh
    
    def record(self, h):
        history = 1 - h.reshape(-1) + self.eps
        self.weights[self.last_idx] = history / np.sum(history)
        
    def get_weights(self):
        a = self.weights[self.last_idx]
        return a + (1 - a.mean())
    
    
class DifficultyWeightedDatasetSampler:
    def __init__(self, dataset, eps=0):
        self.dataset = dataset
        
        self.indices = np.arange(len(self.dataset))
        self.probs = np.ones_like(self.indices) / len(self.indices)
        self.weights = np.ones_like(self.indices) / len(self.indices)
                
        self.last_idx = None
        self.eps = eps
        
    def get_batch(self, batch_size):
        if batch_size > np.sum(self.probs > 0):
            self.probs = np.ones_like(self.indices) / len(self.indices)
            
        self.probs /= np.sum(self.probs)
        idx = np.random.choice(self.indices, size=(batch_size, ), replace=False, p=self.probs)
        self.probs[idx] = 0
        
        data = self.dataset[idx].clone()
        
        self.last_idx = idx
        return data
    
    def record(self, h):
        history = 1 - h.reshape(-1) + self.eps
        self.weights[self.last_idx] = history / np.sum(history)
        
    def get_weights(self):
        a = self.weights[self.last_idx]
        return a + (1 - a.mean())
    
    
class ImportanceConditionalDatasetSampler:
    def __init__(self, dataset, labels, eps=0):
        self.dataset = dataset
        self.labels = labels
        self.num_classes = len(set(labels.tolist()))
        
        self.indices = np.arange(len(self.dataset))
        self.probs = np.ones_like(self.indices) / len(self.indices)
        self.trace = np.ones_like(self.indices) / len(self.indices)
        self.weights = np.ones_like(self.indices) / len(self.indices)
                
        self.last_idx = None
        self.eps = eps
        
    def get_batch(self, batch_size):
        if batch_size > np.sum(self.probs > 0):
            self.probs[self.probs == 0] = self.trace[self.probs == 0]
            self.weights[self.probs == 0] = self.trace[self.probs == 0]
            self.trace = np.ones_like(self.indices) / len(self.indices)
            
        self.probs /= np.sum(self.probs)
        idx = np.random.choice(self.indices, size=(batch_size, ), replace=False, p=self.probs)
        self.probs[idx] = 0
        
        data = self.dataset[idx].clone()
        labels_oh = torch.nn.functional.one_hot(self.labels[idx].clone().long(), self.num_classes).float()
        
        self.last_idx = idx
        return data, labels_oh
    
    def record(self, h):
        h = h.reshape(-1)
        self.trace[self.last_idx] = 1 - h + self.eps
        
    def get_weights(self):
        a = 1/self.weights[self.last_idx]
        return a + (1 - a.mean())
    
    
class ImportanceDatasetSampler:
    def __init__(self, dataset, eps=0):
        self.dataset = dataset
        
        self.indices = np.arange(len(self.dataset))
        self.probs = np.ones_like(self.indices) / len(self.indices)
        self.trace = np.ones_like(self.indices) / len(self.indices)
        self.weights = np.ones_like(self.indices) / len(self.indices)
                
        self.last_idx = None
        self.eps = eps
        
    def get_batch(self, batch_size):
        if batch_size > np.sum(self.probs > 0):
            self.probs[self.probs == 0] = self.trace[self.probs == 0]
            self.weights[self.probs == 0] = self.trace[self.probs == 0]
            self.trace = np.ones_like(self.indices) / len(self.indices)
            
        self.probs /= np.sum(self.probs)
        idx = np.random.choice(self.indices, size=(batch_size, ), replace=False, p=self.probs)
        self.probs[idx] = 0
        
        data = self.dataset[idx].clone()
        self.last_idx = idx
        return data
    
    def record(self, h):
        h = h.reshape(-1)
        self.trace[self.last_idx] = 1 - h + self.eps
        
    def get_weights(self):
        a = 1/self.weights[self.last_idx]
        return a + (1 - a.mean())
    

class EasinessWeightedConditionalDatasetSampler:
    def __init__(self, dataset, labels, eps=0):
        self.dataset = dataset
        self.labels = labels
        self.num_classes = len(set(labels.tolist()))
        
        self.indices = np.arange(len(self.dataset))
        self.probs = np.ones_like(self.indices) / len(self.indices)
        self.weights = np.ones_like(self.indices) / len(self.indices)
                
        self.last_idx = None
        self.eps = eps
        
    def get_batch(self, batch_size):
        if batch_size > np.sum(self.probs > 0):
            self.probs = np.ones_like(self.indices) / len(self.indices)
            
        self.probs /= np.sum(self.probs)
        idx = np.random.choice(self.indices, size=(batch_size, ), replace=False, p=self.probs)
        self.probs[idx] = 0
        
        data = self.dataset[idx].clone()
        labels_oh = torch.nn.functional.one_hot(self.labels[idx].clone().long(), self.num_classes).float()
        
        self.last_idx = idx
        return data, labels_oh
    
    def record(self, h):
        history = h.reshape(-1) + self.eps
        self.weights[self.last_idx] = history / np.sum(history)
        
    def get_weights(self):
        a = self.weights[self.last_idx]
        return a + (1 - a.mean())
    
class EasinessWeightedDatasetSampler:
    def __init__(self, dataset, eps=0):
        self.dataset = dataset
        
        self.indices = np.arange(len(self.dataset))
        self.probs = np.ones_like(self.indices) / len(self.indices)
        self.weights = np.ones_like(self.indices) / len(self.indices)
                
        self.last_idx = None
        self.eps = eps
        
    def get_batch(self, batch_size):
        if batch_size > np.sum(self.probs > 0):
            self.probs = np.ones_like(self.indices) / len(self.indices)
            
        self.probs /= np.sum(self.probs)
        idx = np.random.choice(self.indices, size=(batch_size, ), replace=False, p=self.probs)
        self.probs[idx] = 0
        
        data = self.dataset[idx].clone()
        
        self.last_idx = idx
        return data
    
    def record(self, h):
        history = h.reshape(-1) + self.eps
        self.weights[self.last_idx] = history / np.sum(history)
        
    def get_weights(self):
        a = self.weights[self.last_idx]
        return a + (1 - a.mean())
    
    
class EasinessDatasetSampler:
    def __init__(self, dataset, eps=0):
        self.dataset = dataset
        
        self.indices = np.arange(len(self.dataset))
        self.probs = np.ones_like(self.indices) / len(self.indices)
        self.trace = np.ones_like(self.indices) / len(self.indices)
                
        self.last_idx = None
        self.eps = eps
        
    def get_batch(self, batch_size):
        if batch_size > np.sum(self.probs > 0):
            self.probs[self.probs == 0] = self.trace[self.probs == 0]
            self.trace = np.ones_like(self.indices) / len(self.indices)
            
        self.probs /= np.sum(self.probs)
        idx = np.random.choice(self.indices, size=(batch_size, ), replace=False, p=self.probs)
        self.probs[idx] = 0
        
        data = self.dataset[idx].clone()
        self.last_idx = idx
        
        return data
    
    def record(self, h):
        h = h.reshape(-1)
        self.trace[self.last_idx] = h + self.eps
        
    
class EasinessConditionalDatasetSampler:
    def __init__(self, dataset, labels, eps=0):
        self.dataset = dataset
        self.labels = labels
        self.num_classes = len(set(labels.tolist()))
        
        self.indices = np.arange(len(self.dataset))
        self.probs = np.ones_like(self.indices) / len(self.indices)
        self.trace = np.ones_like(self.indices) / len(self.indices)
                
        self.last_idx = None
        self.eps = eps
        
    def get_batch(self, batch_size):
        if batch_size > np.sum(self.probs > 0):
            self.probs[self.probs == 0] = self.trace[self.probs == 0]
            self.trace = np.ones_like(self.indices) / len(self.indices)
            
        self.probs /= np.sum(self.probs)
        idx = np.random.choice(self.indices, size=(batch_size, ), replace=False, p=self.probs)
        self.probs[idx] = 0
        
        data = self.dataset[idx].clone()
        labels_oh = torch.nn.functional.one_hot(self.labels[idx].clone().long(), self.num_classes).float()
        
        self.last_idx = idx
        return data, labels_oh
    
    def record(self, h):
        h = h.reshape(-1)
        self.trace[self.last_idx] = h + self.eps