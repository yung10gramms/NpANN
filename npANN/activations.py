import numpy as np 

class ActivationFunction():
    def __call__(self, x):
        pass

    def gradient(self, x):    
        pass

class ReLU(ActivationFunction):

    def __call__(self, x):
        return np.maximum(0, x)
        
    def gradient(self, x):
        # return (x >= 0).astype(float)
        if x.ndim == 1:
            return np.diag((x >= 0).astype(float))
        (n, N) = np.shape(x) 
        out = np.zeros((n, n, N))
        for i in range(N):
            out[:,:,i] = np.diag((x[:,i] >= 0).astype(float))
        return out

class Sigmoid(ActivationFunction):

    def __call__(self, x):
        if x.ndim == 1:
            return 1/(1 + np.exp(-x))
        
    def gradient(self, x):
        y = 1/(1 + np.exp(-x))
        if x.ndim == 1:
            return y*(1 - y)
        (n, N) = np.shape(x) 
        out = np.zeros((n, n, N))
        for i in range(N):
            out[:,:,i] = np.diag(y[:,i]*(1 - y[:,i]))
        return out

class Tanh(ActivationFunction):
    def __call__(self, x):
        return np.tanh(x)
    
    def gradient(self, x):
        if x.ndim == 1:
            return 1 - (np.tanh(x))**2
        (n, N) = np.shape(x) 
        out = np.zeros((n, n, N))
        for i in range(N):
            out[:,:,i] = np.diag(1 - (np.tanh(x[:, i]))**2)
        return out
        

class Linear(ActivationFunction):
    def __call__(self, x):
        return x
    
    def gradient(self, x):
        # return np.eye(len(x))
        return np.ones_like(x)


class Softmax(ActivationFunction):
    def __call__(self, x):
        # Subtract max(x) along the feature dimension for numerical stability
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
    
    def gradient(self, x):
        if len(x.shape) <= 1:
            exp_x = np.exp(x)
            softmax_vals = exp_x / np.sum(exp_x)

            n_features = x.shape[0]
            jacobians = np.zeros((n_features, n_features))

            diag = np.diag(softmax_vals)  # Diagonal matrix of softmax values
            outer = np.outer(softmax_vals, softmax_vals)  # Outer product
            jacobians = diag - outer  # Jacobian for the i-th sample

            # print(f'\n\n\n\n\ngradient output shape {np.shape(jacobians)}')E
            return jacobians
        
        
        # Compute softmax probabilities batch-wise
        exp_x = np.exp(x)
        softmax_vals = exp_x / np.sum(exp_x, axis=0, keepdims=True)

        # Initialize the Jacobian for each sample in the batch
        batch_size = x.shape[1]
        n_features = x.shape[0]
        jacobians = np.zeros((n_features, n_features, batch_size))

        # Compute the Jacobian for each sample
        for i in range(batch_size):
            diag = np.diag(softmax_vals[:, i])  # Diagonal matrix of softmax values
            outer = np.outer(softmax_vals[:, i], softmax_vals[:, i])  # Outer product
            jacobians[:,:,i] = diag - outer  # Jacobian for the i-th sample

        # print(f'\n\n\n\n\ngradient output shape {np.shape(jacobians)}')
        return jacobians
"""
unit tests
"""
def run_test():
    x = np.linspace(-10, 10, 100)
    # sig = Sigmoid()
    # sig = Tanh()
    sig = ReLU()
    y = sig(x)
    yp = sig.gradient(x)

    from matplotlib import pyplot as plt

    plt.plot(x, y)
    plt.plot(x, yp)
    plt.show()
