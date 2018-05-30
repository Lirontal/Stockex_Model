#
# # first, we import the relevant modules from the NLTK library
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# import requests
#
# # next, we initialize VADER so we can use it within our Python script
# sid = SentimentIntensityAnalyzer()
# index=0
# index2=0
# r = requests.get('https://api.intrinio.com/news?identifier=TEVA', auth=('2e86cf6af95b890307324803e2de1168', '0f3ff2f93c1a33fd3c1002ade5ba10f8'))
# jsontry=r.json()
# print(jsontry)
# print(jsontry["data"][0]["summary"])
# for x in range(0, 3):
#  # contents = requests.get("https://api.intrinio.com/news?identifier=")
# # the variable 'message_text' now contains the text we will analyze.
#   if r.find("2018-05-28",index) is not -1 : #find if there is n article for the date
#      # index=r.find("2018-05-28")   #move the index to the location
#      # message.text=                #TODO:want to copy the article summary into here (from the start of it to the end)
#       # TODO: send to check the result
#       #TODO: put it in dataframe and check for the previous date
# message_text = '''Like you, I am getting very frustrated with this process. I am genuinely trying to be as reasonable as possible. I am not trying to "hold up" the deal at the last minute. I'm afraid that I am being asked to take a fairly large leap of faith after this company (I don't mean the two of you -- I mean Enron) has screwed me and the people who work for me.'''
#
# #print(message_text)
# print ("1")
# # Calling the polarity_scores method on sid and passing in the message_text outputs a dictionary with negative, neutral, positive, and compound scores for the input text
# scores = sid.polarity_scores(message_text)
#
# # Here we loop through the keys contained in scores (pos, neu, neg, and compound scores) and print the key-value pairs on the screen
#
# for key in sorted(scores):
#         print('{0}: {1}, '.format(key, scores[key]), end='')


# first, we import the relevant modules from the NLTK library
# first, we import the relevant modules from the NLTK library
# first, we import the relevant modules from the NLTK library
# first, we import the relevant modules from the NLTK library
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
import pandas as pd
import numpy as np

# next, we initialize VADER so we can use it within our Python script
sid = SentimentIntensityAnalyzer()
index=0
indexjson=0
counter=0
index2=0
flag=0
indexnext=0
pos=[0,0,0,0]
neg=[0,0,0,0]
counterdates=0
dict= pd.DataFrame(columns=['date', 'compound', 'neg','neu','pos'])
neu=[0,0,0,0]
general=[0,0,0,0]
compound=[0,0,0,0]
r = requests.get('https://api.intrinio.com/news?identifier=TEVA', auth=('2e86cf6af95b890307324803e2de1168', '0f3ff2f93c1a33fd3c1002ade5ba10f8')) #access to json text
jsontry=r.json() #json format
print(jsontry)
fulldate=jsontry["data"][indexjson]["publication_date"]
# the variable 'message_text' now contains the text we will analyze.
while fulldate.find("2018-05-29") is not -1 and indexjson<100: #find if there is an article for the date
  #  index=r.find("2018-05-29")   #move the index to the location
    print(jsontry["data"][indexjson]["publication_date"])
    print(jsontry['data'][indexjson]['summary'])
    counter=counter+1
    #counterdates+=1
    flag=1
    indexjson=indexjson+1
    fulldate = jsontry["data"][indexjson]["publication_date"]

    message_text=jsontry['data'][indexjson]['summary']
    scores = sid.polarity_scores(message_text)
    for key in sorted(scores):
            print('{0}: {1}, '.format(key, scores[key]), end='')
            if (key == "compound"):
                general[0] += scores[key]
                compound[counterdates] += scores[key]
                #print("check")
            if (key == "pos"):
                general[3] += scores[key]
                pos[counterdates] += scores[key]
            if (key == "neg"):
                general[1] += scores[key]
                neg[counterdates] += scores[key]
            if (key == "neu"):
                general[2] += scores[key]
                neu[counterdates] += scores[key]


if(flag==1):
    counterdates+=1
    flag=0
print(compound[indexnext] / counter)
print ("\n")
print(neg[indexnext] / counter)
print(neu[indexnext] / counter)
print(pos[indexnext] / counter)
#dict.loc[counterdates] = jsontry["data"][indexjson]["publication_date"]
#dict.loc[counterdates] = [general[n-1] for n in range(1,6)] #TODO: instead of random send

#print(dict)
    #TODO:sum all the neg\pos... and divide them all by counter

counter=0
indexnext=indexnext+1
print('***********next***************\n\n\n\n\n\n\n\n')
while r.text.find("2018-05-28",index) is not -1 and indexjson<100: #find if there is an article for the date
    print('***********next***************\n')
    index=r.text.find("2018-05-28")   #move the index to the location
    print(jsontry["data"][indexjson]["publication_date"])
    print(jsontry['data'][indexjson]['summary'])
    counter=counter+1
    indexjson=indexjson+1

    message_text=jsontry['data'][indexjson]['summary']
    scores = sid.polarity_scores(message_text)
    for key in sorted(scores):
            print('{0}: {1}, '.format(key, scores[key]), end='')
            #if(key=)
            #print(type(key))
           # print(type("compound"))
            #temp=key
            if(key == "compound"):
                compound[indexnext] += scores[key]
                print("check")
            if (key == "pos"):
                pos[indexnext] += scores[key]
            if (key == "neg"):
                neg[indexnext] += scores[key]
            if (key == "neu"):
                neu[indexnext] += scores[key]

    print(compound[indexnext]/counter)
    print(neg[indexnext]/counter)
    print(neu[indexnext]/counter)
    print(pos[indexnext]/counter)


    #TODO:sum all the neg\pos... and divide them all by counter #almost done just need to divide
    #TODO: check if we are in a diffent date (while is not checking it right)

counter=0
indexnext=indexnext+1
while r.text.find("2018-05-27",index) is not -1 and indexjson<100: #find if there is an article for the date
    index=r.find("2018-05-27")   #move the index to the location
    print(jsontry["data"][indexjson]["publication_date"])
    print(jsontry['data'][indexjson]['summary'])
    counter=counter+1
    indexjson=indexjson+1

    message_text=jsontry['data'][indexjson]['summary']
    scores = sid.polarity_scores(message_text)
    for key in sorted(scores):
            print('{0}: {1}, '.format(key, scores[key]), end='')
    #TODO:sum all the neg\pos... and divide them all by counter

counter=0
indexnext=indexnext+1

while r.text.find("2018-05-26",index) is not -1 and indexjson<100: #find if there is an article for the date
    index=r.text.find("2018-05-26")   #move the index to the location
    print(jsontry["data"][indexjson]["publication_date"])
    print(jsontry['data'][indexjson]['summary'])
    counter=counter+1
    indexjson=indexjson+1

    message_text=jsontry['data'][indexjson]['summary']
    scores = sid.polarity_scores(message_text)
    for key in sorted(scores):
            print('{0}: {1}, '.format(key, scores[key]), end='')
    #TODO:sum all the neg\pos... and divide them all by counter

counter=0
     # message.text=                #TODO:want to copy the article summary into here (from the start of it to the end)
      # TODO: send to check the result
      #TODO: put it in dataframe and check for the previous date
#message_text = '''Like you, I am getting very frustrated with this process. I am genuinely trying to be as reasonable as possible. I am not trying to "hold up" the deal at the last minute. I'm afraid that I am being asked to take a fairly large leap of faith after this company (I don't mean the two of you -- I mean Enron) has screwed me and the people who work for me.'''

#print(message_text)
#    print ("1")
# Calling the polarity_scores method on sid and passing in the message_text outputs a dictionary with negative, neutral, positive, and compound scores for the input text
#scores = sid.polarity_scores(message_text)

# Here we loop through the keys contained in scores (pos, neu, neg, and compound scores) and print the key-value pairs on the screen

#for key in sorted(scores):
#        print('{0}: {1}, '.format(key, scores[key]), end='')
