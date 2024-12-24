import numpy as np 
from numpy import shape, zeros, outer 

# given tensors X, Y, of dimensions n x N and m x N
# return the outer product of each columns
def tensorOuter(X, Y):
    (n, N) = shape(X)
    (m, N) = shape(Y)
     
    prod = zeros((n, m, N))
    for i in range(N):
        prod[:, :, i] = outer(X[:, i], Y[:, i])
    return prod