import numpy as np 

class DataHandler(): 

    __data = None
    _counter = 0

    def __init__(self, dataX, dataY, batch_size = 1):
        assert dataX is not None, 'Data X cannot be None. Please ensure that the variable data is initialized.'
        assert dataY is not None, 'Data Y cannot be None. Please ensure that the variable data is initialized.'
        
        assert isinstance(dataX, np.ndarray) and dataX.ndim == 2, 'Invalid data type of variable data. Please ensure that it is a 2d numpy array.'
        assert isinstance(dataY, np.ndarray) and dataX.ndim == 2, 'Invalid data type of variable data. Please ensure that it is a 2d numpy array.'
    
        self.__dataX = dataX
        self.__dataY = dataY
        
        self.__batch_size = batch_size
        (self.__dimX, self.__num_data) = np.shape(dataX)

        (_, n) = np.shape(dataY)
        assert n == self.__num_data, 'dataX and dataY must have the same cardinality'

        self.reset()


    def reset(self):
        self.__end_flag = False
        self._counter = 0

    def endFlag(self):
        return self.__end_flag

    def hasNext(self):
        return not self.endFlag()

    def nextBatch(self):

        next_batch = self._counter+self.__batch_size
        if next_batch >= self.__num_data:
            next_batch = -1
            self.__end_flag = True
        
        self.__currBatchX = self.__dataX[:, self._counter:next_batch]
        self.__currBatchY = self.__dataY[:, self._counter:next_batch]
        
        self._counter = next_batch
        return (self.__currBatchX, self.__currBatchY)
