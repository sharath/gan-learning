import os
import numpy as np
from sklearn.datasets import make_blobs

def generate_datasets(dataset_dir, plot=False):
    if plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
    os.makedirs(dataset_dir, exist_ok=True)
    np.random.seed(0)

    centers = [[i/3,j/3] for i in range(-2, 3) for j in range(-2, 3)]
    data_train = make_blobs(n_samples=50000, n_features=2, centers=centers, cluster_std=0.005)
    data_test = make_blobs(n_samples=10000, n_features=2, centers=centers, cluster_std=0.005)

    if plot:
        colors = {i:np.random.rand(3,) for i in range(len(centers))}
        plt.scatter(data_train[0][:,0], data_train[0][:, 1], c=[colors[l] for l in data_train[1]], marker='.')
        plt.axis('equal')
        plt.savefig('grid')
        plt.close()
    
    np.save(os.path.join(dataset_dir, 'grid_x_train'), data_train[0])
    np.save(os.path.join(dataset_dir, 'grid_y_train'), data_train[1])
    np.save(os.path.join(dataset_dir, 'grid_x_test'), data_test[0])
    np.save(os.path.join(dataset_dir, 'grid_y_test'), data_test[1])
    
    np.random.seed(0)
    
    r = 0.75
    t = [(i*45)*np.pi/180 for i in range(8)]
    
    centers = [(r*np.cos(i), r*np.sin(i)) for i in t]
    data_train = make_blobs(n_samples=50000, n_features=2, centers=centers, cluster_std=0.005)
    data_test = make_blobs(n_samples=50000, n_features=2, centers=centers, cluster_std=0.005)
    
    
    if plot:
        colors = {i:np.random.rand(3,) for i in range(len(centers))}
        plt.scatter(data_train[0][:,0], data_train[0][:, 1], c=[colors[l] for l in data_train[1]], marker='.')
        plt.axis('equal')
        plt.savefig('circle')
        plt.close()
    
    np.save(os.path.join(dataset_dir, 'circle_x_train'), data_train[0])
    np.save(os.path.join(dataset_dir, 'circle_y_train'), data_train[1])
    np.save(os.path.join(dataset_dir, 'circle_x_test'), data_test[0])
    np.save(os.path.join(dataset_dir, 'circle_y_test'), data_test[1])
    
if __name__ == '__main__':
    generate_datasets(dataset_dir='datasets')
    