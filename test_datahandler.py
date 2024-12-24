import datahandler 
import numpy as np
from numpy.random import randn 

d = 12
N = 1023
data = randn(d, N)

dh = datahandler.DataHandler(data, batch_size=17)
dh.reset()

while not dh.endFlag():
    print(f'current : {np.shape(dh.nextBatch())}')

    



