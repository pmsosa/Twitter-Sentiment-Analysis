from collections import Counter
import string

badbag = Counter()
goodbag = Counter()

raw_input()

exclude = set(string.punctuation)


#Training Phase
good = 0
bad = 0

print "Training..."
while (good <= 3000 and bad <= 3000):
	#print "Good: ",good,"| Bad: ",bad
	rinput = raw_input();
	rinput = rinput.split(",")

	# 0: itemID, 1: Sentiment, 2: SentimentSource, 3: SentimentText
	sentiment = int(rinput[1])
	text = rinput[3].lower()
	text = ''.join(ch for ch in text if ch not in exclude)

	#print "Sentiment", sentiment,
	#print "| Text", text

	if (sentiment == 0): 
		if (bad < 1000):
			for w in text.split():
				w = w.replace(" ","").replace("\n","")
				if (w == ""): break;
				badbag[w] += 1
		bad += 1
	else:
		if (good < 1000):
			for w in text.split():
				w = w.replace(" ","").replace("\n","")
				if (w == ""): break;
				goodbag[w] += 1
		good += 1


print "   Learnt",str(len(goodbag)+len(badbag)),"words."



#Testing Phase
correct = 0
attempts = 0
prediction = 0
i = 0

print "Testing..."
while(i < 5000):
	rinput = raw_input();
	rinput = rinput.split(",")

	# 0: itemID, 1: Sentiment, 2: SentimentSource, 3: SentimentText
	sentiment = int(rinput[1])
	text = rinput[3].lower()
	text = ''.join(ch for ch in text if ch not in exclude)


	#Sentiment Prediction
	gscore = 1000;
	bscore = 1000;
	for w in text.split():
		w = w.replace(" ","").replace("\n","")
		tempgscore = goodbag[w];
		tempbscore = badbag[w];

		if (tempgscore == 0): tempgscore = 1.0
		if (tempbscore == 0): tempbscore = 1.0

		tempgscore = tempgscore
		tempbscore = tempbscore

		#print "--", tempgscore, tempbscore

		gscore *= float(tempgscore)
		bscore *= float(tempbscore)

	gscore = gscore / float(len(badbag))
	bscore = bscore / float(len(goodbag))
	
	#print gscore,bscore

	if (gscore > bscore):
		prediction = 1
	else:
		prediction = 0


	attempts += 1
	if (prediction == sentiment):
		correct += 1

	
	#print "Prediction", prediction, 
	#print "| Sentiment", sentiment,
	#print "| Text", text

	i += 1


print "   Final Results:", correct,"/",attempts,"=", str(correct/float(attempts)*100)+"%"



# #Testing Phase
# while (testing):


# for i in test:
# 	print i
# 	wordbag[i] += 1

# print wordbag


'''
ItemID,Sentiment,SentimentSource,SentimentText
1,0,Sentiment140,                     is so sad for my APL friend.............
2,0,Sentiment140,                   I missed the New Moon trailer...
3,1,Sentiment140,              omg its already 7:30 :O
4,0,Sentiment140,          .. Omgaga. Im sooo  im gunna CRy. I've been at this dentist since 11.. I was suposed 2 just get a crown put on (30mins)...
5,0,Sentiment140,         i think mi bf is cheating on me!!!       T_T
6,0,Sentiment140,         or i just worry too much?        
7,1,Sentiment140,       Juuuuuuuuuuuuuuuuussssst Chillin!!
8,0,Sentiment140,       Sunny Again        Work Tomorrow  :-|       TV Tonight
9,1,Sentiment140,      handed in my uniform today . i miss you already
10,1,Sentiment140,      hmmmm.... i wonder how she my number @-)
11,0,Sentiment140,      I must think about positive..
12,1,Sentiment140,      thanks to all the haters up in my face all day! 112-102
13,0,Sentiment140,      this weekend has sucked so far
14,0,Sentiment140,     jb isnt showing in australia any more!
15,0,Sentiment140,     ok thats it you win.
'''
