import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image
from utils import load_data, print, save_model, load_model
from models import Generator, Discriminator, Classifier
from samplers import *

    
def is_weighted(sampler):
    required = [DifficultyWeightedConditionalDatasetSampler, DifficultyWeightedDatasetSampler, ImportanceConditionalDatasetSampler, ImportanceDatasetSampler, EasinessWeightedConditionalDatasetSampler, EasinessWeightedDatasetSampler]
    for t in required:
        if isinstance(sampler, t):
            return True
    return False

def is_recorded(sampler):
    required = [ImportanceDatasetSampler, ImportanceConditionalDatasetSampler, DifficultyWeightedDatasetSampler, DifficultyWeightedConditionalDatasetSampler, DifficultyConditionalDatasetSampler, DifficultyDatasetSampler, EasinessWeightedConditionalDatasetSampler, EasinessWeightedDatasetSampler, EasinessDatasetSampler, EasinessConditionalDatasetSampler]
    for t in required:
        if isinstance(sampler, t):
            return True
    return False
    

def train(seed=0, dataset='mnist', samplers=(UniformDatasetSampler, UniformLatentSampler),
          latent_dim=100, model_dim=512, device='cuda', conditional=False, learning_rate=2e-4,
          betas=(0, 0.9), batch_size=256, iterations=100000, n_critic=5, objective='gan',
          gp_lambda=1, output_dir='results'):
    
    experiment_name = [seed, dataset, samplers[0].__name__, samplers[1].__name__, latent_dim, model_dim,
                       device, conditional, learning_rate, betas[0], betas[1], batch_size, iterations,
                       n_critic, objective, gp_lambda]
    experiment_name = '_'.join([str(p) for p in experiment_name])
    results_dir = os.path.join(output_dir, experiment_name)
    samples_dir = os.path.join(results_dir, 'samples')
    network_dir = os.path.join(results_dir, 'networks')
    eval_log = os.path.join(results_dir, 'eval.log')
    
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(network_dir, exist_ok=True)
    eval_file = open(eval_log, 'w')
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    data, labels = load_data(dataset)
    data_dim, num_classes = data.shape[1], len(set(labels))        
    
    if conditional:
        test_data, test_labels = load_data(dataset, split='test')
        test_dataset = TensorDataset(torch.tensor(test_data).to(device).float(), torch.tensor(test_labels).to(device).long())
        test_dataloader = DataLoader(test_dataset, batch_size=4096)
    
    data_sampler = samplers[0](torch.tensor(data).float(), torch.tensor(labels).long()) if conditional else samplers[0](torch.tensor(data).float())
    noise_sampler = samplers[1](latent_dim, labels) if conditional else samplers[1](latent_dim)
      
    if conditional:
        G = Generator(latent_dim + num_classes, model_dim, data_dim).to(device)
        D = Discriminator(model_dim, data_dim + num_classes).to(device)
        
        C = Classifier(model_dim, data_dim, num_classes).to(device)
        C.train()
        C_optimizer = optim.Adam(C.parameters(), lr=learning_rate)
        C_crit = nn.CrossEntropyLoss()
    else:
        G = Generator(latent_dim, model_dim, data_dim).to(device)
        D = Discriminator(model_dim, data_dim).to(device)
        
    D.train()
    G.train()
    
    D_optimizer = optim.Adam(D.parameters(), lr=learning_rate, betas=betas)
    G_optimizer = optim.Adam(G.parameters(), lr=learning_rate, betas=betas)
    
    if objective == 'gan':
        fake_target = torch.zeros(batch_size, 1).to(device)
        real_target = torch.ones(batch_size, 1).to(device)
    elif objective == 'wgan':
        grad_target = torch.ones(batch_size, 1).to(device)
    
    fixed_latent_batch = noise_sampler.get_batch(64)
    stats = {'D':[], 'G':[], 'ds':[]}
    
    for it in range(iterations+1):
        # Train Discriminator
        data_batch = data_sampler.get_batch(batch_size)
        latent_batch = noise_sampler.get_batch(batch_size)
        
        if conditional:
            x_real, y_real = data_batch[0].to(device), data_batch[1].to(device)
            real_sample = torch.cat([x_real, y_real], dim=1)
            
            z_fake, y_fake = latent_batch[0].to(device), latent_batch[1].to(device)
            x_fake = G(torch.cat([z_fake, y_fake], dim=1)).detach()
            fake_sample = torch.cat([x_fake, y_fake], dim=1)
            
        else:
            x_real = data_batch.to(device)
            real_sample = x_real
            
            z_fake = latent_batch.to(device)
            x_fake = G(z_fake).detach()
            fake_sample = x_fake
        
        D.zero_grad()
        real_pred = D(real_sample)
        fake_pred = D(fake_sample)
        
        if is_recorded(data_sampler):
            data_sampler.record(real_pred.detach().cpu().numpy())
        
        if objective == 'gan':
            if is_weighted(data_sampler):
                weights = torch.tensor(data_sampler.get_weights()).to(device).float().view(real_pred.shape)
            else:
                weights = torch.ones_like(real_pred).to(device)
                
            D_loss = F.binary_cross_entropy(fake_pred, fake_target).mean() +  (weights * F.binary_cross_entropy(real_pred, real_target)).mean()
            stats['D'].append(D_loss.item())
            
        elif objective == 'wgan':
            alpha = torch.rand(batch_size, 1).expand(real_sample.size()).to(device)
            interpolate = (alpha * real_sample + (1 - alpha) * fake_sample).requires_grad_(True)
            gradients = torch.autograd.grad(outputs=D(interpolate),
                                    inputs=interpolate,
                                    grad_outputs=grad_target,
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
            
            gradient_penalty = (gradients.norm(2, dim=1) - 1).pow(2).mean() * gp_lambda
            
            if is_weighted(data_sampler):
                weights = torch.tensor(data_sampler.get_weights()).to(device).float().view(real_pred.shape)
                real_pred = real_pred * weights
                
            D_loss = fake_pred.mean() - real_pred.mean()
            stats['D'].append(-D_loss.item())
            D_loss += gradient_penalty
    
        D_loss.backward()
        D_optimizer.step()
        
        # Train Generator 
        if it % n_critic == 0:
            G.zero_grad()
            
            latent_batch = noise_sampler.get_batch(batch_size)
            
            if conditional:
                z_fake, y_fake = latent_batch[0].to(device), latent_batch[1].to(device)
                x_fake = G(torch.cat([z_fake, y_fake], dim=1))
                fake_pred = D(torch.cat([x_fake, y_fake], dim=1))
            else:
                z_fake = latent_batch.to(device)
                x_fake = G(z_fake)
                fake_pred = D(x_fake)
            
            if objective == 'gan':
                G_loss = F.binary_cross_entropy(fake_pred, real_target).mean()
                stats['G'].extend([G_loss.item()]*n_critic)
            elif objective == 'wgan':
                G_loss = -fake_pred.mean()
                stats['G'].extend([-G_loss.item()]*n_critic)
            
            G_loss.backward()
            G_optimizer.step()
            
            if conditional:                
                G.eval()
                C.train()
                
                for _ in range(n_critic):
                    latent_batch = noise_sampler.get_batch(batch_size)
                    z_fake, y_fake = latent_batch[0].to(device), latent_batch[1].to(device)
                    x_fake = G(torch.cat([z_fake, y_fake], dim=1)).detach()
        
                    C.zero_grad()
                    C_loss = C_crit(C(x_fake), y_fake.argmax(1))
                    C_loss.backward()
                    C_optimizer.step()
                
                C.eval()
                correct, total = 0.0, 0.0
                for idx, (sample, label) in enumerate(test_dataloader):
                    correct += (C(sample).argmax(1).view(-1) == label).sum()
                    total += sample.shape[0]
                
                stats['ds'].append(correct / total)
                G.train()
        
        if it % 50 == 0:    
            plt.gcf().set_size_inches(5, 5)
            if conditional:
                z_fake, y_fake = fixed_latent_batch[0].to(device), fixed_latent_batch[1].to(device)
                x_fake = G(torch.cat([z_fake, y_fake], dim=1))
            else:
                z_fake = fixed_latent_batch.to(device)
                x_fake = G(z_fake)
                
           
            generated = x_fake.view(64, 1, 28, 28)
            save_image(generated, os.path.join(samples_dir, f'{it}.png'))
            
            plt.plot(stats['G'], label='Generator')
            plt.plot(stats['D'], label='Discriminator')
            plt.legend()
            plt.savefig(os.path.join(results_dir, 'loss.png'))
            plt.close()
            
            if conditional:
                plt.plot(stats['ds'], label='Accuracy')
                plt.legend()
                plt.savefig(os.path.join(results_dir, 'accuracy.png'))
                plt.close()
            
        if it % 10 == 0:
            line = f"{it}\t{stats['D'][-1]:.3f}\t{stats['G'][-1]:.3f}"
            if conditional:
                line += f"\t{stats['ds'][-1]*100:.3f}"
                
            print(line, eval_file)
          
    save_model(G, os.path.join(network_dir, 'G_trained.pth'))
    save_model(D, os.path.join(network_dir, 'D_trained.pth'))
    if conditional:
        save_model(C, os.path.join(network_dir, 'C_trained.pth'))
    eval_file.close()
        

def experiments1(seed, dataset, iterations=100000):
    train(seed=seed, dataset=dataset, objective='gan',  iterations=iterations, conditional=False, samplers=(UniformDatasetSampler, UniformLatentSampler))
    train(seed=seed, dataset=dataset, objective='gan',  iterations=iterations, conditional=False, samplers=(UniformDatasetSampler, NormalLatentSampler))
    train(seed=seed, dataset=dataset, objective='wgan', iterations=iterations, conditional=False, samplers=(UniformDatasetSampler, UniformLatentSampler))
    train(seed=seed, dataset=dataset, objective='wgan', iterations=iterations, conditional=False, samplers=(UniformDatasetSampler, NormalLatentSampler))
    
    train(seed=seed, dataset=dataset, objective='gan',  iterations=iterations, conditional=False, samplers=(ScanUniformDatasetSampler, UniformLatentSampler))
    train(seed=seed, dataset=dataset, objective='gan',  iterations=iterations, conditional=False, samplers=(ScanUniformDatasetSampler, NormalLatentSampler))
    train(seed=seed, dataset=dataset, objective='wgan', iterations=iterations, conditional=False, samplers=(ScanUniformDatasetSampler, UniformLatentSampler))
    train(seed=seed, dataset=dataset, objective='wgan', iterations=iterations, conditional=False, samplers=(ScanUniformDatasetSampler, NormalLatentSampler))
    
    train(seed=seed, dataset=dataset, objective='gan',  iterations=iterations, conditional=False, samplers=(DifficultyDatasetSampler, UniformLatentSampler))
    train(seed=seed, dataset=dataset, objective='gan',  iterations=iterations, conditional=False, samplers=(DifficultyDatasetSampler, NormalLatentSampler))
    train(seed=seed, dataset=dataset, objective='wgan', iterations=iterations, conditional=False, samplers=(DifficultyDatasetSampler, UniformLatentSampler))
    train(seed=seed, dataset=dataset, objective='wgan', iterations=iterations, conditional=False, samplers=(DifficultyDatasetSampler, NormalLatentSampler))
    
    train(seed=seed, dataset=dataset, objective='gan',  iterations=iterations, conditional=False, samplers=(DifficultyWeightedDatasetSampler, UniformLatentSampler))
    train(seed=seed, dataset=dataset, objective='gan',  iterations=iterations, conditional=False, samplers=(DifficultyWeightedDatasetSampler, NormalLatentSampler))
    train(seed=seed, dataset=dataset, objective='wgan', iterations=iterations, conditional=False, samplers=(DifficultyWeightedDatasetSampler, UniformLatentSampler))
    train(seed=seed, dataset=dataset, objective='wgan', iterations=iterations, conditional=False, samplers=(DifficultyWeightedDatasetSampler, NormalLatentSampler))
    
    train(seed=seed, dataset=dataset, objective='gan',  iterations=iterations, conditional=False, samplers=(ImportanceDatasetSampler, UniformLatentSampler))
    train(seed=seed, dataset=dataset, objective='gan',  iterations=iterations, conditional=False, samplers=(ImportanceDatasetSampler, NormalLatentSampler))
    train(seed=seed, dataset=dataset, objective='wgan', iterations=iterations, conditional=False, samplers=(ImportanceDatasetSampler, UniformLatentSampler))
    train(seed=seed, dataset=dataset, objective='wgan', iterations=iterations, conditional=False, samplers=(ImportanceDatasetSampler, NormalLatentSampler))
    
    train(seed=seed, dataset=dataset, objective='gan',  iterations=iterations, conditional=False, samplers=(EasinessWeightedDatasetSampler, UniformLatentSampler))
    train(seed=seed, dataset=dataset, objective='gan',  iterations=iterations, conditional=False, samplers=(EasinessWeightedDatasetSampler, NormalLatentSampler))
    train(seed=seed, dataset=dataset, objective='wgan', iterations=iterations, conditional=False, samplers=(EasinessWeightedDatasetSampler, UniformLatentSampler))
    train(seed=seed, dataset=dataset, objective='wgan', iterations=iterations, conditional=False, samplers=(EasinessWeightedDatasetSampler, NormalLatentSampler))
    
    train(seed=seed, dataset=dataset, objective='gan',  iterations=iterations, conditional=False, samplers=(EasinessDatasetSampler, UniformLatentSampler))
    train(seed=seed, dataset=dataset, objective='gan',  iterations=iterations, conditional=False, samplers=(EasinessDatasetSampler, NormalLatentSampler))
    train(seed=seed, dataset=dataset, objective='wgan', iterations=iterations, conditional=False, samplers=(EasinessDatasetSampler, UniformLatentSampler))
    train(seed=seed, dataset=dataset, objective='wgan', iterations=iterations, conditional=False, samplers=(EasinessDatasetSampler, NormalLatentSampler))
        
def experiments2(seed, dataset):
    train(seed=seed, dataset=dataset, objective='gan', conditional=True, samplers=(UniformConditionalDatasetSampler, UniformConditionalLatentSampler))
    train(seed=seed, dataset=dataset, objective='gan', conditional=True, samplers=(UniformConditionalDatasetSampler, NormalConditionalLatentSampler))
    train(seed=seed, dataset=dataset, objective='wgan', conditional=True, samplers=(UniformConditionalDatasetSampler, UniformConditionalLatentSampler))
    train(seed=seed, dataset=dataset, objective='wgan', conditional=True, samplers=(UniformConditionalDatasetSampler, NormalConditionalLatentSampler))
    
    
def experiments3(seed, dataset):
    train(seed=seed, dataset=dataset, objective='gan', conditional=True, samplers=(ScanUniformConditionalDatasetSampler, UniformConditionalLatentSampler))
    train(seed=seed, dataset=dataset, objective='gan', conditional=True, samplers=(ScanUniformConditionalDatasetSampler, NormalConditionalLatentSampler))
    train(seed=seed, dataset=dataset, objective='wgan', conditional=True, samplers=(ScanUniformConditionalDatasetSampler, UniformConditionalLatentSampler))
    train(seed=seed, dataset=dataset, objective='wgan', conditional=True, samplers=(ScanUniformConditionalDatasetSampler, NormalConditionalLatentSampler))
    
    
def experiments4(seed, dataset):
    train(seed=seed, dataset=dataset, objective='gan', conditional=True, samplers=(DifficultyConditionalDatasetSampler, UniformConditionalLatentSampler))
    train(seed=seed, dataset=dataset, objective='gan', conditional=True, samplers=(DifficultyConditionalDatasetSampler, NormalConditionalLatentSampler))
    train(seed=seed, dataset=dataset, objective='wgan', conditional=True, samplers=(DifficultyConditionalDatasetSampler, UniformConditionalLatentSampler))
    train(seed=seed, dataset=dataset, objective='wgan', conditional=True, samplers=(DifficultyConditionalDatasetSampler, NormalConditionalLatentSampler))

        
def experiments5(seed, dataset):
    train(seed=seed, dataset=dataset, objective='gan', conditional=True, samplers=(DifficultyWeightedConditionalDatasetSampler, UniformConditionalLatentSampler))
    train(seed=seed, dataset=dataset, objective='gan', conditional=True, samplers=(DifficultyWeightedConditionalDatasetSampler, NormalConditionalLatentSampler))
    train(seed=seed, dataset=dataset, objective='wgan', conditional=True, samplers=(DifficultyWeightedConditionalDatasetSampler, UniformConditionalLatentSampler))
    train(seed=seed, dataset=dataset, objective='wgan', conditional=True, samplers=(DifficultyWeightedConditionalDatasetSampler, NormalConditionalLatentSampler))


def experiments6(seed, dataset):
    train(seed=seed, dataset=dataset, objective='gan', conditional=True, samplers=(ImportanceConditionalDatasetSampler, UniformConditionalLatentSampler))
    train(seed=seed, dataset=dataset, objective='gan', conditional=True, samplers=(ImportanceConditionalDatasetSampler, NormalConditionalLatentSampler))
    train(seed=seed, dataset=dataset, objective='wgan', conditional=True, samplers=(ImportanceConditionalDatasetSampler, UniformConditionalLatentSampler))
    train(seed=seed, dataset=dataset, objective='wgan', conditional=True, samplers=(ImportanceConditionalDatasetSampler, NormalConditionalLatentSampler))
    

def experiments7(seed, dataset):
    train(seed=seed, dataset=dataset, objective='gan', conditional=True, samplers=(EasinessWeightedConditionalDatasetSampler, UniformConditionalLatentSampler))
    train(seed=seed, dataset=dataset, objective='gan', conditional=True, samplers=(EasinessWeightedConditionalDatasetSampler, NormalConditionalLatentSampler))
    train(seed=seed, dataset=dataset, objective='wgan', conditional=True, samplers=(EasinessWeightedConditionalDatasetSampler, UniformConditionalLatentSampler))
    train(seed=seed, dataset=dataset, objective='wgan', conditional=True, samplers=(EasinessWeightedConditionalDatasetSampler, NormalConditionalLatentSampler))
    
    
def experiments8(seed, dataset):
    train(seed=seed, dataset=dataset, objective='gan', conditional=True, samplers=(EasinessConditionalDatasetSampler, UniformConditionalLatentSampler))
    train(seed=seed, dataset=dataset, objective='gan', conditional=True, samplers=(EasinessConditionalDatasetSampler, NormalConditionalLatentSampler))
    train(seed=seed, dataset=dataset, objective='wgan', conditional=True, samplers=(EasinessConditionalDatasetSampler, UniformConditionalLatentSampler))
    train(seed=seed, dataset=dataset, objective='wgan', conditional=True, samplers=(EasinessConditionalDatasetSampler, NormalConditionalLatentSampler))


if __name__ == '__main__':
    
    
    train(seed=0, dataset='mnist', objective='wgan', conditional=True, samplers=(ScanUniformConditionalDatasetSampler, UniformConditionalLatentSampler), device='cuda:1')
    #seed = int(sys.argv[1])
    #experiment_id = int(sys.argv[2])
    #
    #eval(f'experiments{experiment_id}({seed}, \'mnist\')')