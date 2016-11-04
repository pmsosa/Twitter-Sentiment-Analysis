# MLP for the IMDB problem
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
seed = 7
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

#print len(X_train),len(X_train[0])

# create the model
model = Sequential()
#print X_train
model.add(Embedding(top_words, 32, input_length=max_words))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


# Fit the model

#Batch Size: How many you are training at the same time.
#
#model = Sequential()
#model.add(Embedding(top_words, 32, input_length=max_words))
#model.add(Flatten())
#model.compile("rmsprop","mse")
#output_array = model.predict(X_train)
#print output_array

model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=1, batch_size=100, verbose=1)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

scores = model.evaluate(X_train, y_train, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))