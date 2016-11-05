from Feedforward_Network import Feedforward_Network
from collections import Counter
import re
import numpy as np

np.random.seed(1)

#Removes symbols from the tweet
def preprocess(tweet):
    return re.sub('[!?.,\'\"\\/*-_\]\[{}()<>#$%^&+=]', '', tweet)
    
# Takes a line from the file and returns the tweet split into a list of strings and its sentiment label
def get_words_and_label(line):
    line = line.strip().split(',', 3)
    label = int(line[1])
    return label, preprocess(line[3]).split(' ')

# Takes a training / testing file and fills up a matrix of inputs X and outputs y and returns X, y
def fill_matrix(file, good_probs, bad_probs, total_tweets):
    X = np.empty([total_tweets, 140])
    y = np.empty([total_tweets, 1])
    for i, line in enumerate(file):
        label, words = get_words_and_label(line)
        y[i][0] = label
        
        for j in range(70):
            if j >= len(words):
                X[i][2*j] = 1
                X[i][2*j+1] = 1
            else:
                X[i][2*j] = good_probs[words[j]]
                X[i][2*j+1] = bad_probs[words[j]]
    return X, y
    

#Set up neural network and training/testing files
twitter_predictor = Feedforward_Network(32*70, 250, 1, 1)

import pickle
with open("emb_train.ser","rb") as f:
    (X_train,y_train) = pickle.load(f)
with open("emb_test.ser","rb") as f:
    (X_test,y_test) = pickle.load(f)

    
    print("Done Loading data. Training model...")
# Train model
twitter_predictor.train(X_train, y_train)

print("Testing model on training set...")

# Making predictions on the training dataset is bad but just want to compare.
predictions_train = twitter_predictor.predict(X_train, repeat = 10)

print("Preprocessing and testing model on testing set...")

#X_test, y_test = fill_matrix(test_file, good_probs, bad_probs, total_tweets_test)
predictions_test = twitter_predictor.predict(X_test, repeat = 10)

print "Calculating errors..."
j = 0
test_error = 0
train_error = 0
while j < len(y_train):
    train_error += np.abs(y_train[j] - predictions_train[j])
    test_error += np.abs(y_test[j] - predictions_test[j])
    j+= 1;
   
train_error =float(train_error)/len(y_train)
test_error =float(test_error)/len(y_test)
 
print "train",train_error
print "test",test_error 
 
#print("Average error on train set: ", np.mean(np.abs(y_train - predictions_train)))
#print("Average error on test set: ", np.mean(np.abs(y_test - predictions_test)))
print(y_test[0:5])




