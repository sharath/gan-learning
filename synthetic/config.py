# vanilla gan parameters
params_gan = {
    'seed': 0,
    
    'latent_dim': 2,
    'model_dim': 512,
    'data_dim': 2,
    'learning_rate': 1e-4,
    'betas': (0.5, 0.9),
    'batch_size': 256,
    'iterations': 40000,
    'device': 'cuda:1',
    
    'n_critic': 5,
    'spec_norm_g': False,
    'spec_norm_d': False,
    'gp_lambda': 10
}

# spectral normalized gan with vanilla objective
params_sngan = {
    'seed': 0,
    
    'latent_dim': 2,
    'model_dim': 512,
    'data_dim': 2,
    'learning_rate': 1e-4,
    'betas': (0.5, 0.9),
    'batch_size': 256,
    'iterations': 40000,
    'device': 'cuda:1',
    
    'n_critic': 5,
    'spec_norm_g': True,
    'spec_norm_d': True,
    'gp_lambda': 10
}

# wasserstein-gan with gradient penalty
params_wgangp = {
    'seed': 0,
    'device': 'cuda:1',
    
    'latent_dim': 2,
    'model_dim': 512,
    'data_dim': 2,
    
    'learning_rate': 1e-4,
    'betas': (0, 0.9),
    'batch_size': 256,
    'iterations': 40000,
    'n_critic': 5,
    
    'spec_norm_g': False,
    'spec_norm_d': False,
    'gp_lambda': 0.1
}
