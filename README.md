# NpANN
A library for Artifical Neural Networks only using Numpy. The library can hanle arbitrarily many layers of any size, all of which run in CPU. The project provides way to -very naturally and very similarly to pytorch- load batches of data, using the `datahandler.py` script, build the model and configure it, while also debugging it with many useful log messages using the script `module.py`. There's also implementations for optimizer in the script `optimizer.py`. As of now, only SGD (with some stepsize rules) and ADAM are implemented.

The two testing scripts are `mnist_test.ipynb` and the `mnist_GAN_specific.ipynb`, for training a classifier and a GAN architecture, respectively. Note that the project is still W.I.P., and it is only intended to be a side project for mostly educational purposes.

![NN diagram](/images%20README/nn_diagram.png)

## Get started 🏁
### Define the model
```python
# instantiate the model
nn = Module()

# Append some layers, defaults to ReLU
for i in range(2):
    nn.appendLayer(100)

# Append a layers with specified activation
nn.appendLayer(2, activation='softmax')
# Set a loss function
nn.setLossFunction(loss.CrossEntropyLoss)
```
### Instantiate an optimizer
e.g. with stochastic gradient descent with dynamic step size rule
```python
opt = optimizer.SGD(nn, dynamic_step=True, weight_normalization=True)
```
Alternatively, ADAM is also implemented
```python
opt = optimizer.ADAM(nn)
```

### Train the model
```python
# Seemlessly define a handler to work with data
batch_size = 50
dataHandler = datahandler.DataHandler(x_train, y_train, batch_size=batch_size)

no_epochs = 20
for j in range(no_epochs):
    # Reset the index at loading data
    dataHandler.reset()
    # Very intuitivelly load the each available batch
    while dataHandler.hasNext():
        (batchX, batchY) = dataHandler.nextBatch()
        # Insert the data by calling the instance, 
        # also equivalent to nn.insert(batchX, batchY)
        nn(batchX, batchY)
        # Feed forward
        nn.forward()
        # Calculate loss explicitly, 
        # although it is calculated internally in backpropagation anyway
        nn.calc_loss()
        
        nn.backward()
        # Take a negative gradient step
        opt.step()
```

## Requirements 📜 
To use the project:
- numpy

To run the unit tests:
- tqdm (to see loading bars)
- Tensorflow (to load mnist)
- matplotlib (to plot the data)

## Technical Details concerning branches  📐 
Recently updated workspace, moving from Python 3.10 to 3.12. This has resulted to having to adapt code to resolve some problems. Namely,
- This branch gets the mnist dataset from `Tensorflow` (instead of `keras` explicitly, like other branches). If you are ok with getting mnist from `keras` directly, consider viewing the branch `branchPy3_10`. The branch in question is not maintained whatsoever, and it is recommended that you work with this branch (`main`), instead.

Also, this branch does not contain the `mnist_GAN.ipynb` file, since the ANN is not trained properly, and hence, it has been removed.

## Notebook to train a classifier 
```
mnist_test.ipynb
```

## Notebook for a basic GAN
### Only learning to produce a specific number
```
mnist_GAN_specific.ipynb
```