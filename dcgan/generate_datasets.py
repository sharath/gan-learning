import os
import numpy as np
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
from torchvision.transforms import Compose, Normalize, ToTensor, Resize


def generate_dataset(dataset_name, train_dataset, test_dataset, transform):
    x_train = np.zeros((len(train_dataset), transform(train_dataset[0][0]).view(-1).shape[0]))
    y_train = np.zeros((len(train_dataset)))
    
    x_test = np.zeros((len(test_dataset), transform(test_dataset[0][0]).view(-1).shape[0]))
    y_test = np.zeros((len(test_dataset)))
    
    for i in range(len(train_dataset)):
        x_train[i] = transform(train_dataset[i][0]).view(-1)
        y_train[i] = train_dataset[i][1]
    
    for i in range(len(test_dataset)):
        x_test[i] = transform(test_dataset[i][0]).view(-1)
        y_test[i] = test_dataset[i][1]
        
    np.save(os.path.join(dataset_dir, f'{dataset_name}_x_train'), x_train)
    np.save(os.path.join(dataset_dir, f'{dataset_name}_y_train'), y_train)
    np.save(os.path.join(dataset_dir, f'{dataset_name}_x_test'), x_test)
    np.save(os.path.join(dataset_dir, f'{dataset_name}_y_test'), y_test)


def generate(dataset_dir='datasets'):
    train_dataset = MNIST(root=dataset_dir, download=True, train=True)
    test_dataset = MNIST(root=dataset_dir, download=True, train=False)
    transform = Compose([Resize(64), ToTensor(), Normalize((0.5, ), (0.5, ))])
    generate_dataset('mnist', train_dataset, test_dataset, transform)
    
    train_dataset = FashionMNIST(root=dataset_dir, download=True, train=True)
    test_dataset = FashionMNIST(root=dataset_dir, download=True, train=False)
    transform = Compose([Resize(64), ToTensor(), Normalize((0.5, ), (0.5, ))])
    generate_dataset('fmnist', train_dataset, test_dataset, transform)
    
    train_dataset = CIFAR10(root=dataset_dir, download=True, train=True)
    test_dataset = CIFAR10(root=dataset_dir, download=True, train=False)
    transform = Compose([Resize(64), ToTensor(), Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    generate_dataset('cifar10', train_dataset, test_dataset, transform)


if __name__ == '__main__':
    generate('datasets')