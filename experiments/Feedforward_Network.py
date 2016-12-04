import numpy as np
import random
import math

#Feedforward neural network with 2 hidden layers
class Feedforward_Network:
    def __init__(self, layer1, layer2, layer3):

        
        self.syn0 = np.random.random((layer1, layer2))
        self.syn1 = np.random.random((layer2, layer3))
        self.biases1 = np.random.random((1, layer2))
        self.biases2 = np.random.random((1, layer3))
    
    def nonlin(self, x, deriv=False):
        if deriv==True:
            return x*(1-x)
        return 1/(1+np.exp(-x))

        
    def forward_prop(self, X):
            l0 = X
            l1 = self.nonlin(np.dot(l0, self.syn0) + self.biases1)
            l2 = self.nonlin(np.dot(l1, self.syn1) + self.biases2)
            return l2
        
    def train(self, X, y, epochs = 1, batch_size = 1, rate = 1):
        
        X = np.asarray(X)
        y = np.asarray(y)
        
        for i in range(epochs):
            X_copy = X
            y_copy = y
            together = zip(X_copy, y_copy)
            random.shuffle(together)
            X_copy, y_copy = zip(*together)
            batch_count = len(y_copy) // batch_size
            X_copy = np.array_split(X_copy, batch_count)
            y_copy = np.array_split(y_copy, batch_count)
            
            for j in range(batch_count):
                l0 = X_copy[j]
                l1 = self.nonlin(np.dot(l0, self.syn0) + self.biases1)
                l2 = self.nonlin(np.dot(l1, self.syn1) + self.biases2)
    
                l2_error = y_copy[j] - l2
                l2_delta = l2_error * self.nonlin(l2, deriv=True)
            
                l1_error = l2_delta.dot(self.syn1.T)
                l1_delta = l1_error * self.nonlin(l1, deriv=True)
            
                self.biases1 += np.sum(l1_delta, axis = 0) * rate
                self.biases2 += np.sum(l2_delta, axis = 0) * rate
                
                self.syn1 += l1.T.dot(l2_delta) * rate
                self.syn0 += l0.T.dot(l1_delta) * rate
                
            if i == 0 or epochs % i == epochs // 10:
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
            
                
    AND_predictor.train(X, y, epochs = 10000, batch_size = 4)
    
    p1 = AND_predictor.predict(np.array([[1, 1]]))
    p2 = AND_predictor.predict(np.array([[1, 0]]))
    p3 = AND_predictor.predict(np.array([[0, 1]]))
    p4 = AND_predictor.predict(np.array([[0, 0]]))
    
    print("1 AND 1 prediction:", p1)
    print("1 AND 0 prediction:", p2)
    print("0 AND 1 prediction:", p3)
    print("0 AND 0 prediction:", p4)
    
    x_sin = np.array([[x] for x in range(100)])
    y_sin = np.array([[(math.sin(x) + 1) / 2] for x in range(10)])
    
    sin_predictor = Feedforward_Network(1, 10, 1)
    
    sin_predictor.train(x_sin, y_sin, epochs = 5000, batch_size = 1, rate = 1)
    
    p1 = sin_predictor.predict(np.array([[1]]))
    p2 = sin_predictor.predict(np.array([[6]]))
    p3 = sin_predictor.predict(np.array([[8]]))
    p4 = sin_predictor.predict(np.array([[3.14]]))
    
    print("sin(1) prediction: {}, actual {}".format(p1[0][0], (math.sin(1) + 1)/2))
    print("sin(6) prediction: {}, actual {}".format(p2[0][0], (math.sin(6) + 1)/2))
    print("sin(8) prediction: {}, actual {}".format(p3[0][0], (math.sin(8) + 1)/2))
    print("sin(3.14) prediction: {}, actual {}".format(p4[0][0], (math.sin(3.14) + 1)/2))