These are a couple of really ugly scripts that where part of a long night of experimenting.

<h3>What it does</h3>
- PrepData.py does 3 main things
    - 1. It reads the Dataset and outputs a dataset in lowercase withouth symbols and punctuation (called "processed_Sentiment Analysis Dataset.csv")
    - 2. It creates a vocabulary Counter ("vocab.ser") and a unique vocabulary where no repetition is allowed ("u_vocab.ser")
    - 3. It creates another dataset ("intdataset_maxed.csv") where the words were replaced with relative occurance to a max number (e.g from Max to 0 how many times has this word appeared)
    - 4. It creates arrays ready to use with the data from the intdataset_max ("1.bad","1.good",etc.)
    - Note that anything *.ser* is serialized, so you can pickle load it to use later.

- keras_nn.py trains and tests a neural network with this data extracted.
    
<h3>Usage</h3>
- Drop the "Sentiment Analysis Dataset.cvs" (name sensitive)
- Run prepdata.py
- Run keras_nn.py
