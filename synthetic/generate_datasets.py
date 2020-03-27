import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import os

dataset_dir = 'datasets'
os.makedirs(dataset_dir, exist_ok=True)

np.random.seed(0)

centers = [[i/3,j/3] for i in range(-2, 3) for j in range(-2, 3)]
data = make_blobs(n_samples=50000, n_features=2, centers=centers, cluster_std=0.02)

colors = {i:np.random.rand(3,) for i in range(len(centers))}
plt.scatter(data[0][:,0], data[0][:, 1], c=[colors[l] for l in data[1]], marker='.')
plt.axis('equal')
plt.savefig('grid')
plt.close()

np.save(os.path.join(dataset_dir, 'grid_x'), data[0])
np.save(os.path.join(dataset_dir, 'grid_y'), data[1])

np.random.seed(0)

r = 0.75
t = [(i*45)*np.pi/180 for i in range(8)]

centers = [(r*np.cos(i), r*np.sin(i)) for i in t]
data = make_blobs(n_samples=50000, n_features=2, centers=centers, cluster_std=0.005)

colors = {i:np.random.rand(3,) for i in range(len(centers))}
plt.scatter(data[0][:,0], data[0][:, 1], c=[colors[l] for l in data[1]], marker='.')
plt.axis('equal')
plt.savefig('circle')
plt.close()

np.save(os.path.join(dataset_dir, 'circle_x'), data[0])
np.save(os.path.join(dataset_dir, 'circle_y'), data[1])

