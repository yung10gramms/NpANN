import numpy as np 

class LossFunction():

    def __call__(self, x, y):
        pass 

    def gradient(self, x, y):
        pass 

class L2Loss(LossFunction):

    def __call__(self, x, y):
        return np.linalg.norm(x-y, 2)
    
    def gradient(self, x, y):
        nm = lambda x0, y0 : (x0 == y0).astype(int)*1 + x0 - y0
        return (x-y)/nm(x0=x, y0=y)
    
class MSELoss(LossFunction):
    def __call__(self, x, y):
        return 1/len(x)*(np.linalg.norm(x-y, 2))**2
    
    def gradient(self, x, y):
        return 2*(x-y)/len(x)
    
class CrossEntropyLoss(LossFunction):
    def __call__(self, x, y):
        return -y @ np.log(x)
    
    def gradient(self, x, y):
        return -y/x


def test_l2():
    l2 = L2Loss()
    x = np.random.randn(10)
    y = np.random.randn(10)
    print(l2(x, y))
    print(l2.gradient(x, y))

    y = x
    print(l2(x, y))
    print(l2.gradient(x, y))
