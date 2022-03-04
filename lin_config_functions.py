import numpy as np


def init_model(model, dataset, num_clients):
    if dataset == "MNIST_4class":
        coefs = np.zeros((num_clients, 4, 784))
        intercepts = np.zeros((num_clients, 4))
        model.coef_ = np.random.rand(4, 784)
        model.intercept_ = np.random.rand(4)
        classes = np.array([0,1,2,3])
        model.classes_ = classes
    elif dataset == "fashion_MNIST":
        coefs = np.zeros((num_clients, 10, 784))
        intercepts = np.zeros((num_clients, 10))
        model.coef_ = np.random.rand(10, 784)
        model.intercept_ = np.random.rand(10)
        classes = np.array([0,1,2,3,4,5,6,7,8,9])
        model.classes_ = classes
    elif dataset == "A2_PCA" or dataset == "3node":
        coefs = np.zeros((num_clients, 1, 784))
        intercepts = np.zeros((num_clients, 1))
        model.coef_ = np.random.rand(1, 100)
        model.intercept_ = np.random.rand(1)
        classes = np.array([0,1])
        model.classes_ = classes
    else: # MNIST 2class
        coefs = np.zeros((num_clients, 1, 784))
        intercepts = np.zeros((num_clients, 1))
        model.coef_ = np.random.rand(1, 784)
        model.intercept_ = np.random.rand(1)
        classes = np.array([0,1])
        model.classes_ = classes
    return model, coefs, intercepts