from numpy.random import shuffle
from numpy import arange

def reshuffle(X, Y):
    '''
    Utility function responsible for shuffling the batch.

    The batch expected shape is (N x batch_size) numpy array

    arguments:
    X : input data
    Y : labels

    arguments:
    X : input shuffled across batch dimension
    Y : labels shuffled across batch dimension
    '''

    NData = len(X[0, :])

    assert len(Y[0, :]) == NData, 'Error: X data and Y data must have the same dimension'

    perm_indices = arange(NData)
    shuffle(perm_indices)
    X = X[:, perm_indices]
    Y = Y[:, perm_indices]
    return X, Y