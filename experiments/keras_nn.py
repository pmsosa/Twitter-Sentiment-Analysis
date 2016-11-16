# MLP for the IMDB problem
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
seed = 0
numpy.random.seed(seed)

# load the dataset but only keep the top n words, zero the rest
top_words = 5001
test_split = 0.33
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=top_words)

#print len(X_train),len(X_train[0])

import pickle
with open("1.good","rb") as f:
    X_train_good = pickle.load(f)
with open("1.bad","rb") as f:
    X_train_bad = pickle.load(f)
with open("2.good","rb") as f:
    X_test_good = pickle.load(f)
with open("2.bad","rb") as f:
    X_test_bad = pickle.load(f)


X_train_bad  = X_train_bad[:2500]
X_train_good = X_train_good[:2500]

X_test_bad   = X_test_bad[:1000]
X_test_good  = X_test_good[:1000]




X_train = numpy.asarray(X_train_good+X_train_bad)
y_train = numpy.asarray([1]*len(X_train_good) + [0]*len(X_train_bad))
X_test  = numpy.asarray(X_test_good+X_test_bad)
y_test  = numpy.asarray([1]*len(X_test_good) + [0]*len(X_test_bad))

max_words = 70
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)


print len(X_train),len(X_train[0]),X_train[0]
print len(X_test),len(X_test[0])
print len(y_train),y_train[0]
print len(y_test),y_test[0]

def keras_nn(X_train,y_train,X_test,y_test,verbose=0,batchsize=100,layersize=250,epoch=1,hidden=1):

    #print numpy.asarray(X_train)[0:5]
    # create the model
    model = Sequential()
    #print X_train
    model.add(Embedding(top_words, 32, input_length=max_words))
    model.add(Flatten())
    model.add(Dense(layersize, activation='relu'))
    for h in range(1,hidden):
        model.add(Dense(layersize, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    if (verbose == 1):
        print(model.summary())

    #Fit the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=epoch, batch_size=batchsize, verbose=verbose)
    # Final evaluation of the model
    score_test = model.evaluate(X_test, y_test, verbose=verbose)

    score_train = model.evaluate(X_train, y_train, verbose=verbose)

    if (verbose):
        print("Accuracy TEST: %.2f%%" % (score_test[1]*100))
        print("Accuracy TRAIN: %.2f%%" % (score_train[1]*100))

    return score_test[1]



###Experiments for Charts
# Batch Size
def exp_batchsize(low,high,i):
    global X_train,y_train,X_test,y_test
    print "Batchsize Experiment..."
    batchsize = low
    output = open("batchsize_exp_we","w+");

    #(X_train,y_train,X_test,y_test) = (X_train[:5000],y_train[:5000],X_test[:2000],y_test[:2000])
    while (batchsize <= high):

        result = keras_nn(X_train,y_train,X_test,y_test,batchsize=batchsize)
        
        output.write(str(batchsize)+" "+str(result)+"\n")
        print batchsize, result;
        batchsize += i
    
    output.close()

# Num of Hidden Layers
def exp_numhidden(low,high,i):
    global X_train,y_train,X_test,y_test
    print "Number of Hidden Layers Experiment..."
    hidden = low
    output = open("numhidden_exp_we","w+");

    #(X_train,y_train,X_test,y_test) = (X_train[:5000],y_train[:5000],X_test[:2000],y_test[:2000])
    while (hidden <= high):
        result = keras_nn(X_train,y_train,X_test,y_test,hidden=hidden)
        output.write(str(hidden)+" "+str(result)+"\n")
        print hidden, result;
        hidden += i
    output.close()


# Size of Hidden Layer
def exp_sizehidden(low,high,i):
    global X_train,y_train,X_test,y_test
    print "Size of Hidden Layer Experiment..."
    layersize = low
    output = open("sizehidden_exp_we","w+");

    #(X_train,y_train,X_test,y_test) = (X_train[:5000],y_train[:5000],X_test[:2000],y_test[:2000])
    while (layersize <= high):
        result = keras_nn(X_train,y_train,X_test,y_test,layersize=layersize)
        output.write(str(layersize)+" "+str(result)+"\n")
        print layersize, result;
        layersize += i
    output.close()

# Epoch
def exp_epoch(low,high,i):
    global X_train,y_train,X_test,y_test
    print "Epoch Experiment..."
    epoch = low
    output = open("epoch_exp_we","w+");

    #(X_train,y_train,X_test,y_test) = (X_train[:5000],y_train[:5000],X_test[:2000],y_test[:2000])
    while (epoch <= high):
        result = keras_nn(X_train,y_train,X_test,y_test,epoch=epoch)
        output.write(str(epoch)+" "+str(result)+"\n")
        print epoch, result;
        epoch += i
    output.close()


#Experiments:
print "Starting experiments..."
print "1. Batch Sizes"
exp_batchsize(1,2201,50)

print "1. Number of Hidden Layers"
exp_numhidden(1,20,1)

print "1. Size of Hidden Layer"
exp_sizehidden(1,2000,100)

print "1. Number of Epochs"
exp_epoch(10,20,1)


######
#PRINT WORD EMBEDDINGS TO USE ON OUR OWN NEURAL NETWORK
#print len(X_train),len(X_train[0])
# model = Sequential()
# model.add(Embedding(top_words, 32, input_length=max_words))
# model.add(Flatten())
# model.compile("rmsprop","mse")
# X_test_emb = model.predict(X_test)
# X_train_emb = model.predict(X_train)

#with open("emb_train.ser","wb") as f:
#    pickle.dump((X_train_emb,y_train),f)

#with open("emb_test.ser","wb") as f:
#    pickle.dump((X_test_emb,y_test),f)