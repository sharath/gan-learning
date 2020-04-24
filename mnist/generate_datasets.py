import os
import numpy as np
from sklearn.datasets import make_blobs
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

def generate_datasets(dataset_dir):

    transform = Compose([ToTensor(), Normalize((0.5, ), (0.5, ))])
    
    dataset = MNIST(root=dataset_dir, download=True, train=True)
    mnist_x_train = np.zeros((len(dataset), transform(dataset[0][0]).view(-1).shape[0]))
    mnist_y_train = np.zeros((len(dataset)))
    
    for i in range(len(dataset)):
        mnist_x_train[i] = transform(dataset[i][0]).view(-1)
        mnist_y_train[i] = dataset[i][1]
        
    np.save(os.path.join(dataset_dir, 'mnist_x_train'), mnist_x_train)
    np.save(os.path.join(dataset_dir, 'mnist_y_train'), mnist_y_train)
    
    dataset = MNIST(root=dataset_dir, download=True, train=False)
    mnist_x_test = np.zeros((len(dataset), transform(dataset[0][0]).view(-1).shape[0]))
    mnist_y_test = np.zeros((len(dataset)))
    
    for i in range(len(dataset)):
        mnist_x_test[i] = transform(dataset[i][0]).view(-1)
        mnist_y_test[i] = dataset[i][1]
        
    np.save(os.path.join(dataset_dir, 'mnist_x_test'), mnist_x_test)
    np.save(os.path.join(dataset_dir, 'mnist_y_test'), mnist_y_test)
    
if __name__ == '__main__':
    generate_datasets(dataset_dir='datasets')
    