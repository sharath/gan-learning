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
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image
from utils import load_data, print, save_model, load_model, save_stats
from models import Generator, Discriminator, Classifier
from samplers import *
from time import time

def train(seed, dataset, samplers=(UniformDatasetSampler, UniformLatentSampler),
          latent_dim=100, model_dim=64, device='cuda', conditional=False, learning_rate=2e-4,
          betas=(0.5, 0.999), batch_size=256, iterations=50000, n_critic=1, objective='gan',
          gp_lambda=10, output_dir='results', plot=False, spec_norm=True):
    
    experiment_name = [seed, dataset, samplers[0].__name__, samplers[1].__name__, latent_dim, model_dim,
                       device, conditional, learning_rate, betas[0], betas[1], batch_size, iterations,
                       n_critic, objective, gp_lambda, plot, spec_norm]
    experiment_name = '_'.join([str(p) for p in experiment_name])
    results_dir = os.path.join(output_dir, experiment_name)
    network_dir = os.path.join(results_dir, 'networks')
    eval_log = os.path.join(results_dir, 'eval.log')

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(network_dir, exist_ok=True)
    
    eval_file = open(eval_log, 'w')
    
    if plot:
        samples_dir = os.path.join(results_dir, 'samples')
        os.makedirs(samples_dir, exist_ok=True)
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    data, labels, data_dim, num_classes = load_data(dataset)
    
    data_sampler = samplers[0](torch.tensor(data).float(), torch.tensor(labels).long()) if conditional else samplers[0](torch.tensor(data).float())
    noise_sampler = samplers[1](latent_dim, labels) if conditional else samplers[1](latent_dim)
      
    if conditional:
        
        test_data, test_labels, _, _ = load_data(dataset, split='test')
        test_dataset = TensorDataset(torch.tensor(test_data).to(device).float(), torch.tensor(test_labels).to(device).long())
        test_dataloader = DataLoader(test_dataset, batch_size=4096)
        
        G = Generator(latent_dim + num_classes, model_dim, data_dim).to(device).train()
        D = Discriminator(model_dim, data_dim, conditional=conditional, spec_norm=spec_norm).to(device).train()
        
        C_real = Classifier(model_dim, data_dim, num_classes).to(device).train()
        C_fake = Classifier(model_dim, data_dim, num_classes).to(device).train()
        C_fake.load_state_dict(deepcopy(C_real.state_dict()))

        C_real_optimizer = optim.Adam(C_real.parameters(), lr=learning_rate)
        C_fake_optimizer = optim.Adam(C_fake.parameters(), lr=learning_rate)
        C_crit = nn.CrossEntropyLoss()
        
    else:
        G = Generator(latent_dim, model_dim, data_dim).to(device).train()
        D = Discriminator(model_dim, data_dim, conditional=conditional, spec_norm=spec_norm).to(device).train()

       
    D_optimizer = optim.Adam(D.parameters(), lr=learning_rate, betas=betas)
    G_optimizer = optim.Adam(G.parameters(), lr=learning_rate, betas=betas)
    
    if objective == 'gan':
        fake_target = torch.zeros(batch_size, 1).to(device)
        real_target = torch.ones(batch_size, 1).to(device)
    elif objective == 'wgan':
        grad_target = torch.ones(batch_size, 1).to(device)
    elif objective == 'hinge':
        bound = torch.zeros(batch_size, 1).to(device)
        sub = torch.ones(batch_size, 1).to(device)

    stats = {'D':[], 'G':[], 'C_it':[], 'C_real':[], 'C_fake':[]}
    if plot:
        fixed_latent_batch = noise_sampler.get_batch(20000)
        sample_figure = plt.figure(num=0, figsize=(5, 5))
        loss_figure = plt.figure(num=1, figsize=(10, 5))
        if conditional:
            accuracy_figure = plt.figure(num=2, figsize=(10, 5))
        
    for it in range(iterations+1):
        # Train Discriminator
        data_batch = data_sampler.get_batch(batch_size)
        latent_batch = noise_sampler.get_batch(batch_size)
        
        if conditional:
            x_real, y_real = data_batch[0].to(device), data_batch[1].to(device)
            real_sample = (x_real, y_real)
            
            z_fake, y_fake = latent_batch[0].to(device), latent_batch[1].to(device)
            x_fake = G(torch.cat([z_fake, y_fake], dim=1)).detach()
            fake_sample = (x_fake, y_fake)
            
        else:
            x_real = data_batch.to(device)
            real_sample = (x_real, None)
            
            z_fake = latent_batch.to(device)
            x_fake = G(z_fake).detach()
            fake_sample = (x_fake, None)
        
        D.zero_grad()
        real_pred = D(*real_sample)
        fake_pred = D(*fake_sample)
        
        if is_recorded(data_sampler):
            data_sampler.record(real_pred.detach().cpu().numpy())
        
        if is_weighted(data_sampler):
            weights = torch.tensor(data_sampler.get_weights()).to(device).float().view(real_pred.shape)
        else:
            weights = torch.ones_like(real_pred).to(device)
                
        if objective == 'gan':
            D_loss = F.binary_cross_entropy(fake_pred, fake_target).mean() +  (weights * F.binary_cross_entropy(real_pred, real_target)).mean()
            stats['D'].append(D_loss.item())
            
        elif objective == 'wgan':
            alpha = torch.rand(batch_size, 1).expand(real_sample[0].size()).to(device)
            interpolate = (alpha * real_sample[0] + (1 - alpha) * fake_sample[0]).requires_grad_(True)
            
            gradients = torch.autograd.grad(outputs=D(interpolate, real_sample[1]),
                                    inputs=interpolate,
                                    grad_outputs=grad_target,
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
            
            gradient_penalty = (gradients.norm(2, dim=1) - 1).pow(2).mean() * gp_lambda
            real_pred = real_pred * weights
                
            D_loss = fake_pred.mean() - real_pred.mean()
            stats['D'].append(-D_loss.item())
            D_loss += gradient_penalty
            
        elif objective == 'hinge':
            D_loss = -(torch.min(real_pred - sub, bound) * weights).mean() - torch.min(-fake_pred - sub, bound).mean()
            stats['D'].append(D_loss.item())
    
        D_loss.backward()
        D_optimizer.step()
                
        # Train Generator
        if it % n_critic == 0:
            G.zero_grad()
            
            latent_batch = noise_sampler.get_batch(batch_size)
            
            if conditional:
                z_fake, y_fake = latent_batch[0].to(device), latent_batch[1].to(device)
                x_fake = G(torch.cat([z_fake, y_fake], dim=1))
                fake_pred = D(x_fake, y_fake)
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
            elif objective == 'hinge':
                G_loss = -fake_pred.mean()
                stats['G'].extend([-G_loss.item()]*n_critic)
            
            G_loss.backward()
            G_optimizer.step()
            
        if conditional:
            # Train fake classifier
            C_fake.train()
                    
            C_fake.zero_grad()
            C_fake_loss = C_crit(C_fake(x_fake.detach()), y_fake.argmax(1))
            C_fake_loss.backward()
            C_fake_optimizer.step()
            
            # Train real classifier
            C_real.train()
            
            C_real.zero_grad()
            C_real_loss = C_crit(C_real(x_real), y_real.argmax(1))
            C_real_loss.backward()
            C_real_optimizer.step()
                           
        if it % 100 == 0:
            C_real.eval()
            C_fake.eval()
            real_correct, fake_correct, total = 0.0, 0.0, 0.0
            for idx, (sample, label) in enumerate(test_dataloader):
                real_correct += (C_real(sample).argmax(1).view(-1) == label).sum()
                fake_correct += (C_fake(sample).argmax(1).view(-1) == label).sum()
                total += sample.shape[0]
                
            stats['C_it'].append(it)
            stats['C_real'].append(real_correct / total)
            stats['C_fake'].append(fake_correct / total)
            
            
            
            line = f"{it}\t{stats['D'][-1]:.3f}\t{stats['G'][-1]:.3f}"
            if conditional:
                line += f"\t{stats['C_real'][-1]*100:.3f}\t{stats['C_fake'][-1]*100:.3f}"
            
            print(line, eval_file)
            
            if plot:
                if conditional:
                    z_fake, y_fake = fixed_latent_batch[0].to(device), fixed_latent_batch[1].to(device)
                    x_fake = G(torch.cat([z_fake, y_fake], dim=1))
                else:
                    z_fake = fixed_latent_batch.to(device)
                    x_fake = G(z_fake)
                    
                generated = x_fake.view(64, data_dim, 64, 64)
                save_image(generated, os.path.join(samples_dir, f'{it}.png'))
                
                plt.figure(1)
                plt.clf()
                plt.plot(stats['G'], label='Generator')
                plt.plot(stats['D'], label='Discriminator')
                plt.legend()
                plt.savefig(os.path.join(results_dir, 'loss.png'))
                
                if conditional:
                    plt.figure(2)
                    plt.clf()
                    plt.plot(stats['C_it'], stats['C_real'], label='Real')
                    plt.plot(stats['C_it'], stats['C_fake'], label='Fake')
                    plt.legend()
                    plt.savefig(os.path.join(results_dir, 'accuracy.png'))
                    
            
          
    save_model(G, os.path.join(network_dir, 'G_trained.pth'))
    save_model(D, os.path.join(network_dir, 'D_trained.pth'))
    save_stats(stats, os.path.join(results_dir, 'stats.pth'))
    if conditional:
        save_model(C_real, os.path.join(network_dir, 'C_real_trained.pth'))
        save_model(C_fake, os.path.join(network_dir, 'C_fake_trained.pth'))
    eval_file.close()
        

def uniform_vs_latent():
    dataset = 'mnist'
    for seed in range(5):
        train(seed=seed, dataset=dataset, objective='gan', conditional=True, samplers=(ScanUniformConditionalDatasetSampler, UniformConditionalLatentSampler))
        train(seed=seed, dataset=dataset, objective='gan', conditional=True, samplers=(ScanUniformConditionalDatasetSampler, NormalConditionalLatentSampler))
        
        train(seed=seed, dataset=dataset, objective='wgan', conditional=True, samplers=(ScanUniformConditionalDatasetSampler, UniformConditionalLatentSampler))
        train(seed=seed, dataset=dataset, objective='wgan', conditional=True, samplers=(ScanUniformConditionalDatasetSampler, NormalConditionalLatentSampler))
        
    
    
def data_sampling_methods():
    dataset = 'mnist'
    for seed in range(5):
        for data_sampler in [UniformConditionalDatasetSampler, ScanUniformConditionalDatasetSampler, DifficultyConditionalDatasetSampler, DifficultyWeightedConditionalDatasetSampler, ImportanceConditionalDatasetSampler, EasinessWeightedConditionalDatasetSampler, EasinessConditionalDatasetSampler]:
            train(seed=seed, dataset=dataset, objective='gan', conditional=True, samplers=(data_sampler, UniformConditionalLatentSampler))
            train(seed=seed, dataset=dataset, objective='gan', conditional=True, samplers=(data_sampler, NormalConditionalLatentSampler))
            
            train(seed=seed, dataset=dataset, objective='wgan', conditional=True, samplers=(data_sampler, UniformConditionalLatentSampler))
            train(seed=seed, dataset=dataset, objective='wgan', conditional=True, samplers=(data_sampler, NormalConditionalLatentSampler))
            
            
if __name__ == '__main__':
    train(seed=0, dataset='mnist', objective='gan', conditional=True, samplers=(ScanUniformConditionalDatasetSampler, UniformConditionalLatentSampler))
    #uniform_vs_latent()
    #data_sampling_methods()
