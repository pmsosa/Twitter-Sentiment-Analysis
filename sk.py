from sklearn.neural_network import MLPClassifier
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
                X[i][2*j] = 0.5
                X[i][2*j+1] = 0.5
            else:
                X[i][2*j] = good_probs[words[j]]
                X[i][2*j+1] = bad_probs[words[j]]
    return X, y
    

#Set up neural network and training/testing files
twitter_predictor = Feedforward_Network(140, 20, 5, 1)
train_file = list(open("data/good10000_1", 'r')) + list(open("data/bad10000_1", 'r'))
test_file = list(open("data/good10000_2", 'r')) + list(open("data/bad10000_2", 'r'))
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
    good_probs[key] = good_count[key] / total_good_words
for key in bad_count:
    bad_probs[key] = bad_count[key] / total_bad_words
    

# Create input and output matrices
X_train, y_train = fill_matrix(train_file, good_probs, bad_probs, total_tweets_train)

print("Done preprocessing data. Training model...")
mlp = MLPClassifier(max_iter=1)
# Train model
#twitter_predictor.train(X_train, y_train)
print(y_train.shape)
mlp.fit(X_train, y_train.ravel())

print("Testing model on training set...")

# Making predictions on the training dataset is bad but just want to compare.
#predictions_train = twitter_predictor.predict(X_train)
predictions_train = mlp.predict(X_train)

print("Preprocessing and testing model on testing set...")

X_test, y_test = fill_matrix(test_file, good_probs, bad_probs, total_tweets_test)
#predictions_test = twitter_predictor.predict(X_test)
predictions_test = mlp.predict(X_test)

print("Average error on train set: ", np.mean(np.abs(y_train - predictions_train)))
print("Average error on test set: ", np.mean(np.abs(y_test - predictions_test)))
print(y_test[0:5])