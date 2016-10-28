import random

raw_input() #delete the Header
random.seed(0)
size = 5000
num = 2

print "randomly extracting ",num," balance dataset of",size,"words..."

while (num != 0):
    goodfile = open("good"+str(num)+".csv","w+")
    badfile  = open("bad"+str(num)+".csv","w+")

    good = 0
    bad = 0
    
    while (good < size/2 or bad < size/2):

        #Drop tweet to create randomness
        #Could potentially cause errors, but our dataset is so huge that the odds of running out of tweets is low.

        if (random.randint(0,1)):
            continue;
            
        

        #print "Good: ",good,"| Bad: ",bad
        rinput = raw_input();
        #rinput = rinput.split(",")[1]
        #print rinput

        # 0: itemID, 1: Sentiment, 2: SentimentSource, 3: SentimentText
        sentiment = int(rinput.split(",")[1])
        #text = rinput[3].lower()
        #text = ''.join(ch for ch in text if ch not in exclude)
        if (sentiment == 0 and bad < size/2):
            badfile.write(rinput+"\n")
            bad += 1;
        elif (sentiment == 1 and good < size/2):
            goodfile.write(rinput+"\n")
            good +=1;

            
        if ((good+bad)%1000 == 0): print num,good,bad,(good+bad)
        
    print "Done with set",num
    num -= 1
