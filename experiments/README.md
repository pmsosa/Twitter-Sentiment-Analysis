These are a couple of really ugly scripts that where part of a long night of experimenting.

<h3>Experiments</h3>
- Word Embedding experiment (_Todo: Make scripts cleaner_)
    - __Files:__ PrepData.py and Keras_nn.py
    - __Aim:__ Builds word embeddings and feeds that into the Keras NN.

- Selected Features
    - __Files:__ feature_extractor.py
    - __Aim:__ Select certain features from tweet and feed into Keras NN.
    


<h2>#1 : Word Embedding Experiment</h2>
- PrepData.py does 4 main things
    - 1. It reads the Dataset and outputs a dataset in lowercase withouth symbols and punctuation (called "processed_Sentiment Analysis Dataset.csv")
    - 2. It creates a vocabulary Counter ("vocab.ser") and a unique vocabulary where no repetition is allowed ("u_vocab.ser")
    - 3. It creates another dataset ("intdataset_maxed.csv") where the words were replaced with relative occurance to a max number (e.g from Max to 0 how many times has this word appeared)
    - 4. It creates arrays ready to use with the data from the intdataset_max ("1.bad","1.good",etc.)
    - Note that anything *.ser* is serialized, so you can pickle load it to use later.

- keras_nn.py trains and tests a neural network with this data extracted.
    
<h3>Usage</h3>
- Drop the "Sentiment Analysis Dataset.cvs" (name sensitive) in this folder
- Run prepdata.py
- Run keras_nn.py

<h2>#2: Feature Selection</h2>
- feature_extractor.py does 3 main things:
    - 1. Go through the dataset and build a new processed dataset where each tweet is described as an array of features.
    - 2. Randomly build some test and training sets from that processed dataset.
    - 3. Run the Keras NN on that dataset

<h3>Usage</h3>
- Drop the "Sentiment Analysis Dataset.cvs" (name sensitive) in this folder
- Run feature_extractor.py

<h3>Features</h3>
The currently tested features:
- [0] = # Chars/ # Words
- [1] = Question marks (?)
- [2] = Exclamation marks (!)
- [3] = Pronouns { i, me, we, us, you, he, him, she, her, it, they, them}
- [4] = Smile :) :D XD
- [5] = Sad :( :< :/
- [6] = URL (https:// or http:// or www. or .com or .gov or .edu .io .org)
- [7] = # of Ellipsis
- [8] = # of Hashtags
- [9] = # of Words


<Rough Results>
- For now, without much tweaking both experiments yielded around 58-65% with 20k training and test sets. 
