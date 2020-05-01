import os
import torch
import numpy as np
import builtins
import warnings
from generate_datasets import generate

num_channels = {
    'mnist': 1,
    'fmnist': 1,
    'cifar10': 3
}

def load_data(dataset, dataset_dir='datasets', split='train'):
    if not os.path.exists(dataset_dir):
        generate(dataset_dir)
    data = np.load(os.path.join(dataset_dir, f'{dataset}_x_{split}.npy'))
    labels = np.load(os.path.join(dataset_dir, f'{dataset}_y_{split}.npy'))
    data_dim = num_channels[dataset]
    num_classes = len(set(labels))
    return data, labels, data_dim, num_classes
    
def gan_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
def xavier_init(m):
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		torch.nn.init.xavier_normal_(m.weight.data, gain=0.02)
	elif classname.find('BatchNorm') != -1:
		torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
		torch.nn.init.constant_(m.bias.data, 0.0)
        
        
def print(output, file=None):
    builtins.print(output)
    if file is None:
        return
    file.write(output+'\n')
    file.flush()
    
    
def save_model(model, filename):
    warnings.filterwarnings('ignore')
    torch.save(model, filename)
    
def save_stats(stats, filename):
    torch.save(stats, filename)
    
def load_model(filename):
    return torch.load(filename)
