"""Problem 3 - Training on MNIST"""
import numpy as np
from mytorch.nn.activations import *
from mytorch.nn.batchnorm import BatchNorm1d
from mytorch.nn.linear import Linear
from mytorch.nn.loss import *
from mytorch.nn.sequential import Sequential
from mytorch.optim.sgd import SGD
from mytorch.tensor import Tensor


# TODO: Import any mytorch packages you need (XELoss, SGD, etc)

# NOTE: Batch size pre-set to 100. Shouldn't need to change.
BATCH_SIZE = 100

def mnist(train_x, train_y, val_x, val_y):
    """Problem 3.1: Initialize objects and start training
    You won't need to call this function yourself.
    (Data is provided by autograder)
    
    Args:
        train_x (np.array): training data (55000, 784) 
        train_y (np.array): training labels (55000,) 
        val_x (np.array): validation data (5000, 784)
        val_y (np.array): validation labels (5000,)
    Returns:
        val_accuracies (list(float)): List of accuracies per validation round
                                      (num_epochs,)
    """
    # TODO: Initialize an MLP, optimizer, and criterion
    #mlp = Sequential(Linear(784, 20), BatchNorm1d(20), ReLU(),Linear(20, 10))
    mlp = Sequential(Linear(784, 20), ReLU(),Linear(20, 10))
    optimizer = SGD(mlp.parameters(), momentum=0.9)
    criterion = CrossEntropyLoss()
    # TODO: Call training routine (make sure to write it below)
    val_accuracies = train(mlp, optimizer, criterion, train_x, train_y, val_x, val_y)
    return val_accuracies

def shuffle_train_data(train_x, train_y):
    n_samples = train_x.shape[0]
    shuffle_indices = np.arange(n_samples)
    np.random.shuffle(shuffle_indices)
    shuffled_train_x = train_x[shuffle_indices]
    shuffled_train_y = train_y[shuffle_indices]
    return shuffled_train_x, shuffled_train_y

def split_data_into_batches(train_x, train_y, batch_size=100): 
    if train_x.shape[0] != train_y.shape[0]:
        raise ValueError("train_x 和 train_y 的样本数量不一致！") 
    total_samples = train_x.shape[0] 
    batch_list = []   
    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        batch_data = train_x[start_idx:end_idx]
        batch_labels = train_y[start_idx:end_idx]    
        batch_list.append( (batch_data, batch_labels) )
    
    return batch_list

def train(model, optimizer, criterion, train_x, train_y, val_x, val_y, num_epochs=10):
    """Problem 3.2: Training routine that runs for `num_epochs` epochs.
    Returns:
        val_accuracies (list): (num_epochs,)
    """
    val_accuracies = []
    model.train()
    for k in range(num_epochs):
        shuffled_train_x, shuffled_train_y = shuffle_train_data(train_x, train_y)
        batches = split_data_into_batches(shuffled_train_x, shuffled_train_y, BATCH_SIZE)
        for i, (batch_data, batch_labels) in enumerate(batches):
            batch_data = Tensor(batch_data)
            batch_labels = Tensor(batch_labels)
            optimizer.zero_grad() # clear any previous gradients
            out = model(batch_data)
            loss = criterion(out, batch_labels)
            loss.backward()
            optimizer.step() # update weights with new gradients
            if i % 100 == 0:
                accuracy = validate(model, val_x, val_y)
                val_accuracies.append(accuracy)
                model.train()
    return val_accuracies

def get_idxs_of_largest_values_per_batch(out):
    return np.argmax(out.data, axis=1)

def compare_sum(a, b):
    assert a.shape == b.shape
    return (a == b).sum()

def validate(model, val_x, val_y):
    """Problem 3.3: Validation routine, tests on val data, scores accuracy
    Relevant Args:
        val_x (np.array): validation data (5000, 784)
        val_y (np.array): validation labels (5000,)
    Returns:
        float: Accuracy = correct / total
    """
    #TODO: implement validation based on pseudocode
    model.eval()
    batches = split_data_into_batches(val_x, val_y, BATCH_SIZE)
    num_correct = 0
    for (batch_data, batch_labels) in batches:
        batch_data = Tensor(batch_data)
        batch_labels = Tensor(batch_labels)
        out = model(batch_data)
        batch_preds = get_idxs_of_largest_values_per_batch(out)
        num_correct += compare_sum(batch_preds, batch_labels.data)
    accuracy = num_correct / len(val_x)
    return accuracy
