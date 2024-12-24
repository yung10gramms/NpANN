from keras.datasets import mnist

import numpy as np

import module
import optimizer 
import loss 
import datahandler


from matplotlib import pyplot as plt
import numpy as np
from numpy import shape

from tqdm import tqdm


# def preprocess_data(x, y, limit):
#     # reshape and normalize input data
#     x = x.reshape(x.shape[0], 28 * 28, 1)
#     x = x.astype("float32") / 255
#     # encode output which is a number in range [0,9] into a vector of size 10
#     # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
#     y = np_utils.to_categorical(y)
#     y = y.reshape(y.shape[0], 10, 1)
#     return x[:limit], y[:limit]

def prepocess_data(x_train, y_train, x_test, y_test, training_size, test_size):
    x_train = x_train.reshape(x_train.shape[0], 28 * 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28 * 28, 1)

    x_train = x_train.T
    x_test = x_test.T

    x_train = x_train[:, :, 0:training_size]
    x_test = x_test[:, :, 0:test_size]

    x_train = x_train[0, :, 0:training_size]
    x_test = x_test[0, :, 0:test_size]

    #  y = np_utils.to_categorical(y)
    # y = y.reshape(y.shape[0], 10, 1)
    



    y_test = y_test[:test_size]
    y_train = y_train[:training_size]
   
    # a = np.linspace(0, 10, 1)
    # A = np.outer(np.ones(training_size), a)
    # Y_train = np.outer(y_train, np.ones(10))
    # equindeces = (Y_train == A).astype(int)
    # Y_train = A.T @ equindeces
    
    
    y_tr = np.zeros((10, training_size))
    y_ts = np.zeros((10, test_size))
    
    y_tr[0, :] = y_train
    y_ts[0, :] = y_test

    y_test = y_ts
    y_train = y_tr
    for i in range(training_size):
        n_tmp = int(y_train[0, i])

        
        y_train[:, i] = np.zeros(10)
        y_train[n_tmp, i] = 1
    print(y_train)

    for i in range(test_size):
        n_tmp = int(y_test[0, i])

        y_test[:, i] = np.zeros(10)
        y_test[n_tmp, i] = 1
    # print(y_test)


 

    return (x_train, y_train, x_test, y_test)


def reshuffle(X, Y):
    NData = len(X)
    perm_indices = np.arange(NData)
    np.random.shuffle(perm_indices)
    X = X[perm_indices]
    Y = Y[perm_indices]
    return X, Y

def flatten_input(x):
    x = x.reshape(x.shape[0], 28 * 28, 1)
    """
    TODO figure this out
    """

    x = x.astype("float32") / 255
    return x


# load MNIST using Keras
# Select 1000 training samples and 20 test samples and appropriate preprocess them
training_size = 400
test_size = 100
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# x_train = flatten_input(x_train)
# x_test = flatten_input(x_test)

# x_train = prepocess_data(x_train, y_train, training_size)
# x_test = prepocess_data(x_test, y_test, test_size)

inDimension = 28*28

(x_train, y_train, x_test, y_test) = prepocess_data(x_train, y_train, x_test, y_test, training_size, test_size)

print(np.shape(x_test))
print(np.shape(x_train))
print(np.shape(y_test))
print(np.shape(y_train))


nn = module.Module()

nn.appendLayer(inDimension)


for i in range(5):
    nn.appendLayer(100) #TODO fix warning

nn.appendLayer(10, activation='softmax')
nn.printShape()
# nn.setLossFunction(loss.L2Loss)
nn.setLossFunction(loss.CrossEntropyLoss)


# opt = optimizer.SGD(nn, learning_rate=0.1, dynamic_step=True)
opt = optimizer.SGD(nn, learning_rate=0.1, dynamic_step=True, weight_normalization=True)
# opt = optimizer.SGD(nn, learning_rate=0.1)

loss_vector = []

wnorm = []

# N = 100

dataHandler = datahandler.DataHandler(x_train, y_train, batch_size=50)

# no_epochs = 10
no_epochs = 10
for j in tqdm(range(no_epochs), desc=f"Processing:"):
    
    
    while dataHandler.hasNext():
        (batchX, batchY) = dataHandler.nextBatch()
        # print(f'batchX shape {np.shape(batchX)}, batchY shape {np.shape(batchY)}')

        nn(batchX, batchY)
        nn.forward()

        # nn.printState()

        nn.calc_loss()

        loss_vector.append(nn.loss)
        
        wnorm.append(sum(np.linalg.norm(w) for w in nn.weights))
        nn.backward()

        opt.step()

from matplotlib import pyplot as plt
if no_epochs < 30:
    plt.semilogy(loss_vector, '^-')
else:
    plt.semilogy(loss_vector, '-')
plt.title('Loss Vector (Semilog Scale)')
plt.xlabel('Epoch')
plt.ylabel('Loss (log scale)')
plt.show()

## test
dataHandler = datahandler.DataHandler(x_test, y_test, batch_size=1)
error_count = 0
validation_count = 0
while dataHandler.hasNext():
    (batchX, batchY) = dataHandler.nextBatch()
    # print(f'batchX shape {np.shape(batchX)}, batchY shape {np.shape(batchY)}')

    nn(batchX, batchY)
    nn.forward()
    
    output = np.argmax(nn.states[-1])

    error_count += (output == batchY).astype(int)
    
    validation_count += 1
    
accuracy = (1 - error_count/validation_count)*100
print(f'Evaluation accuracy: {accuracy}%')
# x_train, y_train = preprocess_data(x_train, y_train, 1000)
# x_test, y_test = preprocess_data(x_test, y_test, 20)



