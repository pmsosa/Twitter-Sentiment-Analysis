#####                           ####
### DONT LOOK AT ME...IM UGLY :( ###
#####                           ####

##  ##   ## ##   ## ##   ##
### ### ### ### ### ### ###
###                     ###
### ################### ###
### #                 # ###
### # ############### # ###
### # #             # # ###
### # #             # # ###
### # ############### # ###
### # #               # ###
### # #  ## ##   ###  # ###
### # #  # # #  #     # ###
### # #  # # #  ####  # ###
### # #  # # #     #  # ###
### # #  # # #  ###   # ###
### #                 # ###
### ################### ###
###                     ###
### ### ### ### ### ### ###
##  ##   ## ##   ## ##   ##

import csv
import random
import string
from collections import Counter
import pickle

random.seed(0)

###Part 1. Clean Processed Data & Save Vocabulary
def build_vocab():
    exclude = set(string.punctuation)
    filename = "Sentiment Analysis Dataset.csv"
    outfilename = "processed_"+filename
    output = open(outfilename,"w+")
    vocabulary = Counter()
    seen = 0

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            seen +=1
            sentiment = line[1]
            tweet = line[3]
            
            words = tweet.split() #Split on whitespace
            
            
            #Some basic preprocessing
            for i in range(0,len(words)):
                if "$" in words[i]: #Cash Values
                    words[i] = "NNCASHVALUE"
                #TODO: ADD MORE SCENARIOS
                
                #Filter
                tempWord = ''.join(ch for ch in words[i] if ch not in exclude)
                tempWord = tempWord.lower()
                words[i] = tempWord
                
                
                
                if (tempWord != ""):
                    vocabulary[tempWord]+=1
                
            while "" in words:
                    words.remove("")
            
            
            towrite = sentiment+","+" ".join(str(x) for x in words)
            output.write(towrite+"\n");
            
            if (seen%100000==0): 
                print seen
                print towrite

    output.close()

    with open("vocab.ser","wb") as f:
        pickle.dump(vocabulary,f)


###Part 2. Make Vocabulary Unique
def create_unique_vocab():

    vocabulary = Counter()
    with open("vocab.ser","rb") as f:
        vocabulary = pickle.load(f)

    print vocabulary.most_common(50)

    #Dirty but whatever
    last = vocabulary.most_common(1)[0][1]+10;
    uniqueVocabulary = vocabulary.most_common()
    for i in range(0,len(uniqueVocabulary)):
        if (uniqueVocabulary[i][1] >= last):
            uniqueVocabulary[i] = (uniqueVocabulary[i][0],last-1)
            #print i,uniqueVocabulary[i][1]
        last = uniqueVocabulary[i][1]

    print last;

    if (last < 0):
        last = -1*last;
        for i in range(0,len(uniqueVocabulary)):
            uniqueVocabulary[i] = (uniqueVocabulary[i][0], uniqueVocabulary[i][1]+last)
            
    print uniqueVocabulary[-1]
    
    uniqueVocabulary = Counter(dict(uniqueVocabulary))
    
    with open("u_vocab.ser","wb") as f:
        pickle.dump(uniqueVocabulary,f)
    print uniqueVocabulary["i"]



###Part 3. Save Int Representation of Tweets
def translate_words_to_freq():
    seen = 0
    uniqueVocabulary = Counter()
    with open("u_vocab.ser","rb") as f:
        uniqueVocabulary = Counter(dict(pickle.load(f)))
    
    max = 5000
    limitVocab = uniqueVocabulary.most_common(max)
    top = int(limitVocab[0][1])
    for i in range(0,len(limitVocab)):
        limitVocab[i] = (limitVocab[i][0],limitVocab[i][1]*max/top)
        print limitVocab[i]
    
    uniqueVocabulary = Counter(dict(limitVocab))
    output2 = open("intdataset_maxed.csv","w+");
    
    bad1 = []
    bad2 = []
    good1 = []
    good2 = []
    

    
    with open("processed_Sentiment Analysis Dataset.csv","r") as f:
        reader = csv.reader(f)
        for line in reader:
            seen +=1
            sentiment = line[0]
            tweet = line[1]
            
            temptweet = ""
            for t in tweet.split():
                if temptweet == "":
                    temptweet += str(uniqueVocabulary[t])
                else:
                    temptweet += "."+str(uniqueVocabulary[t])
            
            output2.write(sentiment+","+temptweet+"\n")
            
            
            ####UNCESSESARY SHENNAIGAN
            if (sentiment == "0"):
                if (len(bad1) < 10000):
                    bad1 += [[int(i) for i in temptweet.split(".")] + [0]*(70-len(temptweet.split(".")))]
                    
                elif (len(bad2) < 10000):       
                    bad2 += [[int(i) for i in temptweet.split(".")] + [0]*(70-len(temptweet.split(".")))]
            else:
                if (len(good1) < 10000):
                    good1 += [[int(i) for i in temptweet.split(".")] + [0]*(70-len(temptweet.split(".")))]
                elif (len(good2) < 10000):
                    good2 += [[int(i) for i in temptweet.split(".")] + [0]*(70-len(temptweet.split(".")))]
        
        
        with open("1.bad","wb") as f:
            pickle.dump(bad1,f)
        with open("2.bad","wb") as f:
            pickle.dump(bad2,f)
        with open("1.good","wb") as f:
            pickle.dump(good1,f)
        with open("2.good","wb") as f:
            pickle.dump(good2,f)
                
    output2.close()

build_vocab();
create_unique_vocab();
translate_words_to_freq()