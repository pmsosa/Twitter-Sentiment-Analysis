import numpy as np
import random

random.seed(1)
np.random.seed(1)



def preprocess(tweet):
    tweet = re.sub('[!?.,\'\"\\/*-_\]\[{}()<>#$%^&+=]', '', tweet)
    return tweet.lower()



good = list(open("data/good1.csv", 'r'))
bad = list(open("data/bad1.csv", 'r'))

good = ["I like you,0", "this is awesome,0", "i like this,0", "rainbows are cool,0", "this is funny,0"]
bad  = ["I hate you,1", "dang this place,1", "f*ck this sh*t,1", "god dang this,1", "f*ck you dude,1", "this is silly,1"]

training_set = good + bad

dictionary = []


def lookup(sentence):
    global dictionary

    sentence = sentence.split()
    wordvect = [ ]

    for word in sentence:
        if (word not in dictionary):
            dictionary += [word]

        wordvect += [dictionary.index(word)]

    #print sentence, wordvect

    return np.array([wordvect])


def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    print -x;
    return 1/(1+np.exp(-x))



syn0 = 2*np.random.random((3,100)) - 1
syn1 = 2*np.random.random((100,50)) -1
syn2 = 2*np.random.random((50,1)) - 1

for j in range(0,1000000):
    
    x = random.choice(training_set)
    
    inputs = lookup(x.split(",")[0])
    y = int(x.split(",")[1])


    l0 = inputs
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))
    l3 = nonlin(np.dot(l2,syn2))

    l3_error = y - l3

    if (j% 10000) == 0:
        print "Error:" + str(np.mean(np.abs(l3_error)))

        #print syn0
        #print syn1



    l3_delta = l3_error*nonlin(l3,deriv=True)

    l2_error = l3_delta.dot(syn2.T)
    l2_delta = l2_error*nonlin(l2,deriv=True)

    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error* nonlin(l1,deriv=True)

    syn2 += l2.T.dot(l3_delta)
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

print syn0
print syn1
    

