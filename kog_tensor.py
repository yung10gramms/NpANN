import numpy as np  


# def h():
#     def __call__(self, x):
#         return 

class ReLU():
    def __call__(self, x):
        out = np.zeros_like(x)
        for i, _ in enumerate(out):
            if x[i] > 0:
                out[i] = x[i]
        return out
        # return np.max(0, x)
  
    def grad(self, x):
        grad_out = np.zeros(np.shape(x))
        for i in range(len(grad_out)):
            if x[i] > 0:
                grad_out[i] = 1
        
        return grad_out
    
"""
function for tensor Kroneker product between a matrix (A) and a vector (b)
with the resulting product being C[i, j, :] = A[i, j] * b
"""
def tensor_kprod(A, b):
    (n, m) = np.shape(A)
    k = len(b)
    C = np.zeros((n, m, k))

    for i in range(n):
        for j in range(m):
            C[i, j, :] = A[i, j] * b

    return C

    # shape_a = np.shape(A)
    # shape_b = np.shape(b)
    
    # C = np.zeros((np.unwrap(shape_a), np.unwrap(shape_b)))

    # for a in shape_a:
    #     C[a, :] = A[a]*b
        

A = np.random.random((10, 8))
b = np.random.random(4)

C = tensor_kprod(A, b)
print(np.shape(C))

# relu : ReLU
relu = ReLU()
x = np.random.random(10)-0.5
print(x)
print(relu(x))
print(relu.grad(x))
