import numpy as np
import random

class Feedforward_Network:
    def __init__(self, input_nodes, h_nodes1, h_nodes2, output_nodes):
        self.syn0 = 2*np.random.random((input_nodes, h_nodes1)) - 1
        self.syn1 = 2*np.random.random((h_nodes1, h_nodes2)) - 1
        self.syn2 = 2*np.random.random((h_nodes2, output_nodes)) - 1
        
    def nonlin(self, x, deriv=False):
        if deriv==True:
            return x*(1-x)
        return 1/(1+np.exp(-x))
        
    def forward_prop(self, X):
            l0 = X
            l1 = self.nonlin(np.dot(l0, self.syn0))
            l2 = self.nonlin(np.dot(l1, self.syn1))
            l3 = self.nonlin(np.dot(l2, self.syn2))
            return l3
        
    def train(self, X, y, repeat = 1):
        for j in range(0, repeat):
            l0 = X
            l1 = self.nonlin(np.dot(l0, self.syn0))
            l2 = self.nonlin(np.dot(l1, self.syn1))
            l3 = self.nonlin(np.dot(l2, self.syn2))
        
            l3_error = y - l3
        
            if j % 5000 == 0:
                print("Error:" + str(np.mean(np.abs(l3_error))))
                #print syn0
                #print syn1
        
            l3_delta = l3_error*self.nonlin(l3,deriv=True)
        
            l2_error = l3_delta.dot(self.syn2.T)
            l2_delta = l2_error*self.nonlin(l2,deriv=True)
        
            l1_error = l2_delta.dot(self.syn1.T)
            l1_delta = l1_error* self.nonlin(l1,deriv=True)
        
            self.syn2 += l2.T.dot(l3_delta)
            self.syn1 += l1.T.dot(l2_delta)
            self.syn0 += l0.T.dot(l1_delta)
            
    def predict(self, X):
        return self.forward_prop(X)
    
    
if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)
    
    neural_net = Feedforward_Network(2, 3, 3, 1)
    
    X = np.array([[1, 1],
        [1, 0],
        [0, 1],
        [0, 0]])
        
    y = np.array([[1],
                [0],
                [0],
                [0]])
                
    neural_net.train(X, y, 50000)
    p1 = neural_net.predict(np.array([[1, 1]]))
    p2 = neural_net.predict(np.array([[1, 0]]))
    p3 = neural_net.predict(np.array([[0, 1]]))
    p4 = neural_net.predict(np.array([[0, 0]]))
    
    print("1 AND 1 prediction:", p1)
    print("1 AND 0 prediction:", p2)
    print("0 AND 1 prediction:", p3)
    print("0 AND 0 prediction:", p4)