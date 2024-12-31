# NpANN
A library for Artifical Neural Networks only using Numpy. The library can hanle arbitrarily many layers of any size, which all without parallelization in CPU. The project provides way to very naturally and very similarly to pytorch load batches of data, using the `datahandler.py` script, build the model and configure it, while also debugging it with many useful log messages using the script `module.py`. There's also implementations for optimizer in the script `optimizer.py`. As of now, only SGD (with some stepsize rules) and ADAM are implemented.

The two testing scripts are `mnist_test.ipynb` and the `mnist_GAN.ipynb`, for training a classifier and a GAN architecture, respectively. Note that the project is still W.I.P., and it is only intended to be a side project for mostly educational purposes.

![NN diagram](/images%20README/nn_diagram.png)

## Get started
### Define the model
```python
# instantiate the model
nn = Module()

# Append some layers, defaults to ReLU
for i in range(2):
    discriminator.appendLayer(100)

# Append a layers with specified activation
discriminator.appendLayer(2, activation='softmax')
# Set a loss function
discriminator.setLossFunction(loss.CrossEntropyLoss)
```
### Instantiate an optimizer
e.g. with stochastic gradient descent with dynamic step size rule
```python
dis_opt = optimizer.SGD(discriminator, dynamic_step=True, weight_normalization=True)
```
Alternatively, ADAM is also implemented
```python
opt = optimizer.ADAM(nn)
```
## Requirements
To use the project:
- numpy

To run the unit tests:
- tqdm (to see loading bars)
- keras (to load mnist)
- matplotlib (to plot the data)

## Notebook to train a classifier
```
mnist_test.ipynb
```

## Notebook for a basic GAN
```
mnist_gan.ipynb
```