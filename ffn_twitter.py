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
    line = line.strip().split(',')
    label = int(line[1])
    return label, preprocess(line[3]).split(' ')

#Set up neural network and training/testing files
twitter_predictor = Feedforward_Network(140, 20, 15, 1)
train_file = list(open("data/1000a", 'r'))
test_file = list(open("data/1000b", 'r'))

#Set up counters for good sentiment and bad sentiment categories
bad_count = Counter()
good_count = Counter()
total_tweets = 0
total_good_tweets = 0
total_bad_tweets = 0

#Add words to counters
for line in train_file:
    total_tweets += 1
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
for key in good_count:
    good_count[key] /= total_good_words
for key in bad_count:
    bad_count[key] /= total_bad_words
    

# Create input and output matrices
    
    
    
X = np.empty([total_tweets, 140])
y = np.empty([total_tweets, 1])
    
# Fill up training matrix
for i, line in enumerate(train_file):
    label, words = get_words_and_label(line)
    y[i][0] = label
    
    for j in range(70):
        if j >= len(words):
            X[i][2*j] = 1
            X[i][2*j+1] = 1
        else:
            X[i][2*j] = good_count[words[j]]
            X[i][2*j+1] = bad_count[words[j]]
            

# Train model
twitter_predictor.train(X, y, 10)

# Making predictions on the training dataset is bad but I'm just making sure everything is working. Will implement testing dataset soon.
predictions= twitter_predictor.predict(X)
print(total_good_tweets, total_bad_tweets)
