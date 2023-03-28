import os
from torchvision.datasets import MNIST
import numpy as np

def get_data_from_torchvision_datasets():
    # Prepare MNIST-10 dataset from torchvision.datasets
    train_dataset = MNIST(os.getcwd(), train=True, download=True)
    test_dataset = MNIST(os.getcwd(), train=False, download=True)

    # transform to numpy 
    train_data = np.array(train_dataset.data)
    train_label = np.array(train_dataset.targets)
    test_data = np.array(test_dataset.data)
    test_label = np.array(test_dataset.targets)

    return train_data, train_label, test_data, test_label

