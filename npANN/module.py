import numpy as np 
# from .activations import * 
import activations

from typing import List, Any

class Module():
    weights : List[Any]
    biases  : List[Any]
    layers  : List[Any]
    activs  : List[Any]
    states  : List[Any]

    input   : Any
    output  : Any
    loss    : Any


    total_weights : int
    total_biases  : int
    total_layers  : int

    loss_function : Any 
    loss_gradient : Any 

    RELU = 'relu'
    SIGMOID = 'sigmoid'
    TANH = 'tanh'
    LINEAR = 'linear'
    SOFTMAX = 'softmax'

    valid_activations = {RELU : activations.ReLU, SIGMOID : activations.Sigmoid, TANH : activations.Tanh, LINEAR : activations.Linear, SOFTMAX : activations.Softmax}
    valid_initializations = {'gaussian', 'none', 'uniform', 'xavier'}

    def __init__(self):
        self.weights = []
        self.biases = []
        self.layers = []
        self.activs = []
        self.states = []

        self.total_layers = 0
        self.total_biases = 0
        self.total_weights = 0

        self.input = None
        self.output = None
        self.loss = Any
        self.loss_function = None
        self.loss_gradient = None

        # current label
        self.yhat = None

        self.propagated_gradients = [None]


    def __call__(self, x, y):
        return self.insert(x, y)
    

    
    def setLossFunction(self, loss):
        """
        TODO check the validity of each loss function
        """
        self.loss_instance = loss()
        self.loss_function = lambda x,y : self.loss_instance(x,y)
        self.loss_gradient = lambda x,y : self.loss_instance.gradient(x,y)

    def isEmpty(self):
        return len(self.layers) == 0

    def __getWeightInit(self, initialization, **kwargs):
        assert initialization in self.valid_initializations, f'ERROR: {initialization} not a valid initialization'

        if initialization == 'gaussian':
            return lambda x : np.random.randn(*x) if isinstance(x, tuple) else np.random.randn(x)
        if initialization == 'none':
            return np.zeros
        if initialization == 'uniform':
            return lambda x : np.random.rand(*x) if isinstance(x, tuple) else np.random.rand(x)
        if initialization == 'xavier':
            assert 'n_in' in kwargs and 'n_out' in kwargs, 'Range not specified for Xavier initialization'
            n_in = int(kwargs['n_in'])
            n_out = int(kwargs['n_out'])
            sq = np.sqrt(6/(n_in + n_out))
            return lambda x : (np.random.rand(*x)-0.5)*2*sq if isinstance(x, tuple) else (np.random.rand(x)-0.5)*2*sq
        return 

    def appendLayer(self, n, initialization = 'xavier', activation = None):
        '''
        Function that appends a layer at the end of the model. The layer will be fully connected to the previous one. 

        arguments:
        n                 : size of the layer
        initialization    : initialization to apply to the layer. Defaults to Xavier
        activation        : activation function for the layer. Defaults to ReLU
        '''
        assert initialization in self.valid_initializations, f'ERROR: {initialization} not a valid initialization'

        self.total_layers += 1

        is_empty = self.isEmpty()
        if is_empty and activation is not None:
            print('Warning: input layer cannot have activation, input ignored')

        if not is_empty and activation is None:
            activation = self.RELU

        if not is_empty: 
            self.activs.append(activation)
        else:
            self.activs.append(None)

        self.layers.append(n)
        self.states.append(np.zeros(n))

        if is_empty:
            # no need for weights with just one layer
            return
        
        if initialization == 'xavier':
            kw = {'n_in': self.layers[-2], 'n_out': self.layers[-1]}
            init_func = self.__getWeightInit(initialization, **kw)    
        else:
            init_func = self.__getWeightInit(initialization)
        
        self.total_weights += self.layers[-1]*self.layers[-2]
        self.total_biases += self.layers[-1] 

        self.weights.append(init_func((self.layers[-1], self.layers[-2])))
        self.biases.append((init_func(self.layers[-1])))

        self.propagated_gradients.append(None)

    def insert(self, x, y):
        '''
        Insert an input-label pair to the model. Equivalent to calling the instance.
        '''
        assert not self.isEmpty(), 'Cannot pass input into empty Module'
        # TODO fix these assertions
        # assert np.shape(x) == np.shape(self.states[0]), f"Invalid shape of input : is {np.shape(x)} and expected {np.shape(self.states[0])}"
        # assert np.shape(y) == np.shape(self.states[-1]), f"Invalid shape of output : is {np.shape(y)} and expected {np.shape(self.states[-1])}"

        self.input = x
        
        self.yhat = y


    def forward(self):
        '''
        Forward the input, calculating all the values of all states.
        '''
        x = self.input

        for i, _ in enumerate(self.states):
            if i == 0:
                self.states[0] = x
                continue
            activation_class = self.valid_activations[self.activs[i]]
            activation_tmp = activation_class()
            if x.ndim == 1:
                self.states[i] = activation_tmp(self.weights[i-1] @ self.states[i-1] + self.biases[i-1])
            elif x.ndim == 2: 
                b_vect = self.biases[i-1]
                self.states[i] = activation_tmp(self.weights[i-1] @ self.states[i-1] + b_vect[:, np.newaxis])
                

        self.output = self.states[-1]
        return 

    def calc_loss(self):
        '''
        Calculate the value of the loss function. If working with batches, the function will calculate the average across inputs.
        '''
        assert self.yhat is not None, 'Label y cannot be none. Make sure you forward() before you call loss_function'
        if self.output.ndim == 1:
            self.loss = self.loss_function(self.output, self.yhat)
        
            return self.loss
        
        (_, batch) = np.shape(self.output)
        
        self.loss = 1/batch*np.sum(self.loss_function(self.output[:, j], self.yhat[:, j]) for j in range(batch))

        return self.loss

    def backward(self):
        '''
        Apply backpropagation algorithm on the model.
        '''

        assert self.total_layers > 1, f'Module too small to backpropagate.\nNum layers: {self.total_layers}.\nExpected: >1'
        
        logs_flag = False

        g_map = [None]*(self.total_layers)
        z_map = [None]*len(self.biases)

        self.gradsW = [None]*len(self.weights)
        self.gradsb = [None]*len(self.biases)
        
        
        self.calc_loss()

        g_map[-1] = self.loss_gradient(self.output, self.yhat)

        if logs_flag:
            print(f' number of weights {len(self.weights)}, number of layers {self.total_layers}, number of states {len(self.states)}')

            print(f'shape of output={np.shape(self.output)} shape of yhat={np.shape(self.yhat)}')
            print(f'g^(k+1) shape = {np.shape(g_map[-1])}')

        #TODO check for out of bounds
        for i in range(self.total_layers-2, -1, -1):
            
            """
            Not so clean method to seperate the cases whether we are dealing with batches or not
            """
            if self.states[i].ndim == 1:
                z_map[i] = self.weights[i] @ self.states[i] + self.biases[i]
            else:
                b_vect = self.biases[i]
                z_map[i] = self.weights[i] @ self.states[i] + b_vect[:, np.newaxis]

            if logs_flag:
                print(f'-----------------------------------------------------------\niteration at index {i}')
                
                print('zmap ', np.shape(z_map[i]))

        

            if self.activs[i+1] is None:
                print(f'Warning: no activation function at layer {i}. Function will exit\n')
                return
            
            actClass = self.valid_activations[self.activs[i+1]]
            actInst = actClass()

            sigma_prime_z =  actInst.gradient(np.squeeze(z_map[i])) 

            """
            TODO fix this problem: find a cleaner method to handle jacobians
            """
            if g_map[i+1].ndim > 1:
                
                nabla_L_XY = g_map[i+1]
                a = np.zeros_like(nabla_L_XY)
                # Compute batch matrix-vector products
                for n in range(len(nabla_L_XY[0,:])):
                    a[:, n] = np.dot(sigma_prime_z[:, :, n], nabla_L_XY[:, n])

            else:
                a = sigma_prime_z * g_map[i+1]
            
            if logs_flag:
                print(f'sigma_prime_z = {np.shape(sigma_prime_z)}')
                print(f'a = sigma_prime_z * g_map[i+1] = {np.shape(a)}')
                

            if self.states[i].ndim > 1:
                X = self.states[i]
                
                (_, N) = np.shape(a)
                self.gradsW[i] = a @ X.T/N 

            else:
                self.gradsW[i] = np.outer(a, self.states[i])

            
            self.gradsb[i] = a
            if self.gradsb[i].ndim == 2:
                self.gradsb[i] = np.apply_along_axis(np.average, 1, self.gradsb[i])


            g_map[i] =  self.weights[i].T @ a
            
            # Store propagated gradient into a member variable
            self.propagated_gradients[i] = g_map[i]

            if logs_flag:
                

                print(f'SHAPE OF gradW: {np.shape(self.gradsW[i])}')
                print(f'SHAPE OF gradb: {np.shape(self.gradsb[i])}')
                
                print(f'g map new dimensions {np.shape(g_map[i])}')

        return
    
    def updateWeight(self, i, newVal):
        self.weights[i] = newVal
        
    def incrWeight(self, i, incr):
        self.weights[i] += incr

    def normalizeWeight(self, i):
        nW = np.linalg.norm(self.weights[i])
        if nW == 0: # this should not happen
            return
        self.weights[i] /= nW

    def updateBias(self, i, newVal):
        self.biases[i] = newVal
    
    def incrBias(self, i, incr):
        self.biases[i] += incr

    def normalizeBias(self, i):
        nb = np.linalg.norm(self.biases[i])
        if nb == 0: # this should not happen
            return
        self.biases[i] /= nb

    def __getNNsize(self, size_bytes):
        lgn = np.log10(size_bytes)
        idx = np.floor(lgn/3)
        
        if lgn < 3:
            return ('', size_bytes)
        if lgn < 6:
            return ('K', size_bytes/1024)
        if lgn < 9:
            return ('M', size_bytes/1024**2)
        if lgn < 12:
            return ('G', size_bytes/1024**3)
        return ('T', size_bytes/1024**4)


    def printShape(self, readable=True):
        '''
        function that prints the shape of the neural network, including the Weights dimensions
        at each layer, the number of weights and biases, as well as an approximation of the model size in RAM.

        arguments:
        readable    : if set to false it will print the size of the NN in Kbytes. Else, it will print it in a human-readable way. Defaults to True.      
        '''
        print(f'Number of Layers: {self.total_layers}\nTotal Parameters: {self.total_biases+self.total_weights} ({self.total_weights} weights and {self.total_biases} biases)')
        byte_size = 8*(self.total_biases+self.total_weights)
        if not readable:
            print(f'Size on RAM (approximate) {byte_size/1024:.2f} kb')
        else:
            st, sz = self.__getNNsize(byte_size)
            print(f'Size on RAM (approximate) {sz:.2f} {st}b')
        print(' x '.join([f'({str(layer)} {str(self.activs[i])})' for i, layer in enumerate(self.layers)]))
        print('Weight dimensions: '+', '.join([f'{str(np.shape(w))}' for w in self.weights]))
        
    def printState(self):
        for i, state in enumerate(self.states):
            print(f'layer {i+1}: state {state}')
