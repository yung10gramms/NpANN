import numpy as np 
import module 

class Optimizer(): 
    def __init__(self, m : module.Module): 
        pass 

    def step(self):
        pass

class ADAM(Optimizer):

    def __init__(self, m : module.Module, alpha = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8):
        self.module = m
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        ############################
        self.parameters = dict()
        print('ADAM initialized')

        

    class Inner__params__():
        def __init__(self):
            self.m_t = 0
            self.v_t = 0
            self.g_t = 0
            self.t = 0

    def __make_adam_step(self, cur_objective, grad_tmp):
        cur_objective.t += 1
        cur_objective.g_t = grad_tmp
        cur_objective.m_t = self.beta_1*cur_objective.m_t + (1 - self.beta_1)*cur_objective.g_t
        
        cur_objective.v_t = self.beta_2*cur_objective.v_t + (1 - self.beta_2)*(cur_objective.g_t**2)
        m_t_hat = cur_objective.m_t/(1 - self.beta_1**cur_objective.t)
        v_t_hat = cur_objective.v_t/(1 - self.beta_2**cur_objective.t)
        return - self.alpha*m_t_hat/(np.sqrt(v_t_hat) + self.epsilon)
    
    def step(self):
        # print('step adam')
        for i, _ in enumerate(self.module.weights):
            gradW = self.module.gradsW[i]
            gradb = self.module.gradsb[i]
            
            
            # Adam algorithm for weights
            if 'weight'+str(i) not in self.parameters:
                self.parameters['weight'+str(i)] = self.Inner__params__()
            
            cur_objective = self.parameters['weight'+str(i)]
            
            adam_step_weights =self.__make_adam_step(cur_objective, gradW)
            # print(f'adam step norm {np.linalg.norm(adam_step_weights)}')
            self.module.incrWeight(i, adam_step_weights)

            # Adam algorithm for biases
            if 'bias'+str(i) not in self.parameters:
                self.parameters['bias'+str(i)] = self.Inner__params__()
            
            cur_objective = self.parameters['bias'+str(i)]
            self.module.incrBias(i, self.__make_adam_step(cur_objective, gradb))
            


class SGD(Optimizer):
    __learning_rate_default = 0.9

    def __init_learning_rate(self, learning_rate, dynamic_step):
        self.learning_rate = learning_rate 
        self.dynamic_step = dynamic_step
        if dynamic_step and (learning_rate is not None):
            print('Warning: dynamic step option will ignore learning rate')
            self.k = 1
            self.learning_rate = None
            return
        elif not dynamic_step and (learning_rate is None):
            self.learning_rate = self.__learning_rate_default
            return
        

    def __init__(self, m: module.Module, learning_rate = None, dynamic_step = False, weight_normalization = False): 
        '''
        Initialize Stochastic Gradient Descent Algorithm.

        arguments:
        m                    : module
        learning_rate        : learning rate, defaults to 0.9
        dynamic_step         : flag to apply dynamic step rule
        weight_normalization : flag to determine whether weights are going to be normalized
        '''
        super().__init__(m)
        self.module = m
        
        self.__init_learning_rate(learning_rate, dynamic_step)

        self.weight_normalization = weight_normalization

    
    def step(self):
        noZero = 0
        for i, _ in enumerate(self.module.weights):
            gradW = self.module.gradsW[i]
            gradb = self.module.gradsb[i]

            # print('shape of gradient W ', np.shape(gradW))

            if self.dynamic_step:
                if  np.linalg.norm(gradW) == 0:
                    # print('Warning: zero gradient')
                    self.learning_rate = 0
                else:
                    self.learning_rate = 1/(self.k * np.linalg.norm(gradW))
                   
            self.module.incrWeight(i,-self.learning_rate*gradW)
            if self.weight_normalization:
                self.module.normalizeWeight(i)

            if sum([np.linalg.norm(w) for w in self.learning_rate*gradW]) == 0:
                # print('zero grad')
                noZero += 1

            if self.dynamic_step:
                if  np.linalg.norm(gradb) == 0:
                    # print('Warning: zero gradient')
                    self.learning_rate = 0
                else:
                    self.learning_rate = 1/(self.k * np.linalg.norm(gradb))
                # self.learning_rate = 1/(self.k * np.linalg.norm(gradb))
                self.k += 1

            
            self.module.incrBias(i, -self.learning_rate*gradb)
            if self.weight_normalization:
                self.module.normalizeBias(i)
            

        

        
    