# Feature Extractor
# 
# Features:
#   - [0] = # Chars/ # Words
#   - [1] = Question marks (?)
#   - [2] = Exclamation marks (!)
#   - [3] = Pronouns { i, me, we, us, you, he, him, she, her, it, they, them}
#   - [4] = Smile :) :D XD
#   - [5] = Sad :( :< :/
#   - [6] = URL (https:// or http:// or www. or .com or .gov or .edu .io .org)
#   - [7] = # of Ellipsis
#   - [8] = # of Hashtags

import csv
import random
import pickle

import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

random.seed(0)

def extract_features():
    feat_num     = 10

    chwd         = 0
    questions    = 1
    exclamations = 2
    pronoun      = 3
    smile        = 4
    sad          = 5
    url          = 6
    ellipsis     = 7
    hashtags     = 8
    capitals     = 9
    #length       = 10

    pronouns = ["i","me", "we", "us", "you", "he", "him", "she", "her", "it", "they", "them"]
    good_emoticons = [":)", ":D", "XD",":P", ":p", ";)",";D",";P"]
    bad_emoticons = [":(", ":S", ":'(", ">:(", ":/"]

    output = open("featured_dataset.csv","w+")
    filename = "Sentiment Analysis Dataset.csv"

    seen = 0

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        
        for line in reader:
            seen += 1
            sentiment = line[1]
            tweet = line[3]
            stweet = tweet.split()

            features = [0] * feat_num

            #0. Char/Words
            features[chwd] = float(len(tweet))/len(stweet)

            #10. Length
            #features[length] = len(stweet)

            for word in stweet:

                #1. Question Marks
                if "?" in word:
                    features[questions] += 1

                #2. Exclamation Marks
                if "!" in word:
                    features[exclamations] += 1

                #7. Ellipsis Marks
                if "..." in word:
                    features[ellipsis] += 1

                #6. URLs
                if "https://" in word.lower():
                    features[url] += 1
                elif "https://" in word.lower():
                    features[url] += 1
                elif "www." in word.lower():
                    features[url] += 1
                elif ".com" in word:
                    features[url] += 1

                #8. Hashtags
                if word[0] == "#":
                    features[hashtags] += 1

                #4. Good Emoticon
                if word in good_emoticons:
                    features[smile] += 1
                #5. Bad Emoticon
                elif word in bad_emoticons:
                    features[sad] += 1

                else:
                    #3. Pronouns
                    for p in pronouns:
                        if p == word.lower():
                            features[pronoun] += 1
                            break;
                #9. Capital Letters
                for ch in word:
                    features[capitals] += int(ch.isupper())

            #Save 
            towrite = sentiment

            for f in features:
                towrite += "," + str(f)

            output.write(towrite + "\n")
            if (seen %10000 == 0): print seen,"---",towrite

    output.close()



def create_batches(size=2000,split=0.5):

    X_train     = []
    y_train     = []
    train_good  = 0
    train_bad   = 0

    X_test     = []
    y_test     = []
    test_good  = 0
    test_bad   = 0

    with open("featured_dataset.csv", 'r') as f:
        reader = csv.reader(f)



        for line in reader:
            sentiment = line[0]
            tweet = line[1:]

            if (test_good == test_bad == train_good == train_bad == size*split): break;

            #Randomly drop some
            if (random.randint(0,1)):
                
                #Randomly drop into Train or Test set
                if (random.randint(0,1)):
                    if (int(sentiment) == 1): 
                        if (train_good >= size*split): continue;
                        train_good += 1
                    else: 
                        if (train_bad >= size*split): continue;
                        train_bad += 1

                    X_train += [[float(i) for i in tweet]]
                    y_train += [int(sentiment)]


                else:

                    if (int(sentiment) == 1): 
                        if (test_good >= size*split): continue;
                        test_good += 1
                    else: 
                        if (test_bad >= size*split): continue;
                        test_bad += 1

                    X_test += [[float(i) for i in tweet]]
                    y_test += [int(sentiment)]


            print (train_bad,train_good,test_bad,test_good,size*split)


    print len(X_train), len(y_train)
    print len(X_test), len(y_test)
    #Serialize
    with open("train.ser","wb") as f:
        pickle.dump((X_train,y_train),f)

    with open("test.ser","wb") as f:
        pickle.dump((X_test,y_test),f)

    return (X_train,y_train,X_test,y_test)



def keras_nn(X_train,y_train,X_test,y_test):

    print numpy.asarray(X_train)[0:5]
    # create the model
    model = Sequential()
    #print X_train
    model.add(Dense(100, activation='relu',input_dim=10))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())


    # Fit the model

    #Batch Size: How many you are training at the same time.
    #
    model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=1, batch_size=256, verbose=1)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    scores = model.evaluate(X_train, y_train, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))


###MAIN

extract_features();
(X_train,y_train,X_test,y_test) = create_batches(20000,0.5)

keras_nn(X_train,y_train,X_test,y_test)

#65-61
#61 without length