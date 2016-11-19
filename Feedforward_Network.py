import numpy as np
import random

#Feedforward neural network with 2 hidden layers
class Feedforward_Network:
    def __init__(self, layer1, layer2, layer3):
        self.syn0 = 2*np.random.random((layer1, layer2)) - 1
        self.syn1 = 2*np.random.random((layer2, layer3)) - 1
    
    def nonlin(self, x, deriv=False):
        if deriv==True:
            return x*(1-x)
        return 1/(1+np.exp(-x))

        
    def forward_prop(self, X):
            l0 = X
            l1 = self.nonlin(np.dot(l0, self.syn0))
            l2 = self.nonlin(np.dot(l1, self.syn1))
            return l2
        
    def train(self, X, y, epochs = 1, batch_size = 1):
        
        X = np.asarray(X)
        y = np.asarray(y)
        
        for i in range(epochs):
            X_copy = X
            y_copy = y
            together = zip(X_copy, y_copy)
            random.shuffle(together)
            X_copy, y_copy = zip(*together)
            X_copy = np.array_split(X_copy, batch_size)
            y_copy = np.array_split(y_copy, batch_size)
            
            for j in range(batch_size):
                l0 = X_copy[j]
                l1 = self.nonlin(np.dot(l0, self.syn0))
                l2 = self.nonlin(np.dot(l1, self.syn1))
    
                l2_error = y_copy[j] - l2
                l2_delta = l2_error*self.nonlin(l2,deriv=True)
            
                l1_error = l2_delta.dot(self.syn1.T)
                l1_delta = l1_error* self.nonlin(l1,deriv=True)
            
                
                self.syn1 += l1.T.dot(l2_delta)
                self.syn0 += l0.T.dot(l1_delta)
                
            if batch_size // 10 != 0 and (i == 0 or i % (batch_size // 10) == 0):
                print("Error:" + str(np.mean(np.abs(l2_error))))
                #print syn0
                #print syn1
                
            
    def predict(self, X):
        return self.forward_prop(X)
    
    
if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)
    
    AND_predictor = Feedforward_Network(2, 30, 1)
    
    X = np.array([[1, 1],
        [1, 0],
        [0, 1],
        [0, 0]])
        
    y = np.array([[1],
                [0],
                [0],
                [0]])
                
    AND_predictor.train(X, y, epochs = 10000, batch_size = 1)
    
    p1 = AND_predictor.predict(np.array([[1, 1]]))
    p2 = AND_predictor.predict(np.array([[1, 0]]))
    p3 = AND_predictor.predict(np.array([[0, 1]]))
    p4 = AND_predictor.predict(np.array([[0, 0]]))
    
    print("1 AND 1 prediction:", p1)
    print("1 AND 0 prediction:", p2)
    print("0 AND 1 prediction:", p3)
    print("0 AND 0 prediction:", p4)