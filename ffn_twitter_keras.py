#from Feedforward_Network import Feedforward_Network
from collections import Counter
import re
import numpy as np
import random

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

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
                X[i][j] = 0
                X[i][2*j+1] = 0
            else:
                X[i][j] = good_probs[words[j]] * 100
                X[i][2*j+1] = bad_probs[words[j]] *100
    return X, y
    

#Set up neural network and training/testing files
#twitter_predictor = Feedforward_Network(70, 250, 1, 1)
train_file = list(open("data/good10000_1", 'r')) + list(open("data/bad10000_1", 'r'))
random.shuffle(train_file)
test_file = list(open("data/good10000_2", 'r')) + list(open("data/bad10000_2", 'r'))
random.shuffle(test_file)
#print(train_file, test_file)

#Set up counters for good sentiment and bad sentiment categories
bad_count = Counter()
good_count = Counter()
total_tweets_train = len(train_file)
total_tweets_test = len(test_file)
total_good_tweets = 0
total_bad_tweets = 0

#Add words to counters
for line in train_file:
    label, words = get_words_and_label(line)
    
    if label:
        total_good_tweets += 1
        count_tracker = good_count
    else:
        total_bad_tweets += 1
        count_tracker = bad_count
        
    for w in words:
        count_tracker[w] += 1

total_good_words = len(good_count)
total_bad_words = len(bad_count)


#Convert word counts to probabilities
good_probs = Counter()
bad_probs = Counter()

for key in good_count:
    good_probs[key] = good_count[key] / float(total_good_words)
for key in bad_count:
    bad_probs[key] = bad_count[key] / float(total_bad_words)
    

# Create input and output matrices
X_train, y_train = fill_matrix(train_file, good_probs, bad_probs, total_tweets_train)

#print("Done preprocessing data. Training model...")

# Train model
#twitter_predictor.train(X_train, y_train,42)

#print("Testing model on training set...")

# Making predictions on the training dataset is bad but just want to compare.
#predictions_train = twitter_predictor.predict(X_train)

#print("Preprocessing and testing model on testing set...")

X_test, y_test = fill_matrix(test_file, good_probs, bad_probs, total_tweets_test)
#predictions_test = twitter_predictor.predict(X_test)

#print("Average error on train set: ", np.mean(np.abs(y_train - predictions_train)))
#print("Average error on test set: ", np.mean(np.abs(y_test - predictions_test)))
#print(y_test[0:5])


print "KERAS EXPERIMENT ----"
model = Sequential()
model.add(Dense(250,input_dim=140,activation="relu"))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=2, batch_size=128, verbose=1)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy (TEST): %.2f%%" % (scores[1]*100))

scores = model.evaluate(X_train, y_train, verbose=0)
print("Accuracy (TRAIN): %.2f%%" % (scores[1]*100))


