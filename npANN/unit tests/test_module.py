import module as module 
import optimizer as optimizer 
import loss 

from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

import datahandler as datahandler


def test_batch():
    nn = module.Module()

    nn.appendLayer(2)

    # for i in range(2):
    #     # nn.appendLayer(15, activation='sigmoid')
    #     # nn.appendLayer(15, activation='linear')
    #     nn.appendLayer(5) #TODO fix warning

    for i in range(5):
        nn.appendLayer(10) #TODO fix warning

    nn.appendLayer(1, activation='linear')
    nn.printShape()
    # nn.setLossFunction(loss.L2Loss)
    nn.setLossFunction(loss.MSELoss)

    # nn.printState()


    # y_tmp = np.random.randn(7)

    eigs = np.random.rand(2)
    L = np.diag(eigs)
    Q = np.random.rand(2, 2)
    Amatrix = Q.T @ L @ Q

    b = np.random.rand(2)
    c = np.random.rand(1)+1

    # f = lambda x0 : np.array(x0.T @ Amatrix @ x0 + x0 @ b + c )
    # f = lambda x0 : np.array(x0.T @ x0 + x0 @ b + c )
    f = lambda x0 : np.array(np.log(x0.T @ Amatrix @ x0 + c) )

    # opt = optimizer.SGD(nn, learning_rate=0.1, dynamic_step=True)
    opt = optimizer.SGD(nn, learning_rate=0.1, dynamic_step=True, weight_normalization=True)
    # opt = optimizer.SGD(nn, learning_rate=0.1)

    loss_vector = []

    wnorm = []

    N = 1000
    dataX = np.random.randn(2, N)
    dataY = np.apply_along_axis(f, 0, dataX)

    print(f'X shape {np.shape(dataX)}, Y shape {np.shape(dataY)}')
    dataHandler = datahandler.DataHandler(dataX, dataY, batch_size=50)

    # no_epochs = 10
    no_epochs = 1000
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
    plots_flag = True
    if plots_flag :
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        # Plot 1: Loss vector in linear scale
        axs[0, 0].plot(loss_vector, '-')
        axs[0, 0].set_title('Loss Vector (Linear Scale)')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('Loss')

        # Plot 2: Loss vector in semilog scale
        if no_epochs < 30:
            axs[0, 1].semilogy(loss_vector, '^-')
        else:
            axs[0, 1].semilogy(loss_vector, '-')
        axs[0, 1].set_title('Loss Vector (Semilog Scale)')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('Loss (log scale)')

        # Plot 3: Loss vector in log-log scale
        axs[1, 0].loglog(loss_vector, '-')
        axs[1, 0].set_title('Loss Vector (Log-Log Scale)')
        axs[1, 0].set_xlabel('Epoch (log scale)')
        axs[1, 0].set_ylabel('Loss (log scale)')

        # Plot 4: Weight norms in semilogy scale
        axs[1, 1].semilogy(wnorm, '-')
        axs[1, 1].set_title('Weight Norms (Semilog Scale)')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].set_ylabel('Weight Norm (log scale)')

        # Adjust layout
        plt.tight_layout()

        # Show the plots
        plt.show()

    
def test_simple():

    nn = module.Module()

    nn.appendLayer(2)

    for i in range(2):
        # nn.appendLayer(15, activation='sigmoid')
        # nn.appendLayer(15, activation='linear')
        nn.appendLayer(5) #TODO fix warning

    nn.appendLayer(1, activation='linear')
    nn.printShape()

    # nn.setLossFunction(loss.L2Loss)
    nn.setLossFunction(loss.MSELoss)

    eigs = np.random.rand(2)
    L = np.diag(eigs)
    Q = np.random.rand(2, 2)
    Amatrix = Q.T @ L @ Q

    b = np.random.rand(2)
    c = np.random.rand(1)+1

    # f = lambda x0 : np.array(x0.T @ Amatrix @ x0 + x0 @ b + c )
    # f = lambda x0 : np.array(x0.T @ x0 + x0 @ b + c )
    f = lambda x0 : np.array(np.log(x0.T @ Amatrix @ x0) )

    opt = optimizer.SGD(nn, learning_rate=0.1, dynamic_step=True)
    # opt = optimizer.SGD(nn, learning_rate=0.1)

    loss_vector = []

    x_xor = [[0,0], [0,1], [1,0], [1,1]]
    y_xor = [0, 1, 1, 0]

    wnorm = []

    for j in tqdm(range(10000), desc="Processing"):
        # print(f'iternum{j}')
        x_tmp = np.random.randn(2)
        y_tmp = f(x_tmp)

        nn.insert(x_tmp, y_tmp)
        nn.forward()

        # nn.printState()

        nn.calc_loss()
        loss_vector.append(nn.loss)
        
        wnorm.append(sum(np.linalg.norm(w) for w in nn.weights))
        nn.backward()

        opt.step()

    plots_flag = True
    if plots_flag :
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        # Plot 1: Loss vector in linear scale
        axs[0, 0].plot(loss_vector, '-')
        axs[0, 0].set_title('Loss Vector (Linear Scale)')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('Loss')

        # Plot 2: Loss vector in semilog scale
        axs[0, 1].semilogy(loss_vector, '-')
        axs[0, 1].set_title('Loss Vector (Semilog Scale)')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('Loss (log scale)')

        # Plot 3: Loss vector in log-log scale
        axs[1, 0].loglog(loss_vector, '-')
        axs[1, 0].set_title('Loss Vector (Log-Log Scale)')
        axs[1, 0].set_xlabel('Epoch (log scale)')
        axs[1, 0].set_ylabel('Loss (log scale)')

        # Plot 4: Weight norms in semilogy scale
        axs[1, 1].semilogy(wnorm, '-')
        axs[1, 1].set_title('Weight Norms (Semilog Scale)')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].set_ylabel('Weight Norm (log scale)')

        # Adjust layout
        plt.tight_layout()

        # Show the plots
        plt.show()


# test_simple()

test_batch()