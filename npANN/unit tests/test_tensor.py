import numpy as np 
from numpy.random import randn 
import activations 


def test_tensor():
    d_l = 3
    d_l1 = 6
    N = 1
    A = randn(d_l, d_l1)
    b = randn(d_l)
    # B = b[:, np.newaxis]
    # B = b[:, N]
    # X = randn(d_l1, N)
    X = randn(d_l1, N)
    Z = A @ X + b[:, np.newaxis]

    print(f"prod is {Z}")

    act = activations.ReLU()
    Y = act(Z)
    print(Y)

    Sp = act.gradient(Z)
    print(Sp)
     

# test_tensor()