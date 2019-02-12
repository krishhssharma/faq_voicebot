'''This program convert the questions and answers from voice to text form in a single file. Then we cluster the texts into 3 clusters. 
   If users ask a FAQ in voice form, it gives the text answer of the query from the respected cluster.'''

import nltk
import json
import warnings
warnings.filterwarnings("ignore")
#nltk.download() # for downloading packages

import speech_recognition as sr
from gtts import gTTS

import re
import numpy as np
import random
import string # to process standard python strings

import math
import os
from random import sample
import _pickle as cPickle  #The pickle module already imports _pickle if available. It is the C-optimized version of the pickle module, and is used transparently.
from tqdm import *         #for progress bar

from scipy.spatial.distance import cdist  #Compute distance between each pair of the two collections of inputs.
from texttable import Texttable   #Python module for creating simple ASCII tables
import indicoio             #A Python Wrapper for indico. Use pre-built state of the art machine learning algorithms with a single line of code.

indicoio.config.api_key = "b1b35f65d921fd2d6d0f1933e3f9e99a"

from textblob import TextBlob    #TextBlob is a python library and offers a simple API to access its methods and perform basic NLP tasks. 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import numpy


n_similar_val=0
faqs = {}
with open('C:/Users/krish/Desktop/Data Science/b3ds/speech/faq_voice_bot/ipfile.txt') as f:
    for line in f:
        ln=line.split(':')
        n_similar_val=len(ln)+1
        x=ln[0]
        y=ln[1]
        z=len(y)-1
        y=y[0:z]
        faqs[(x)]=y

def calculate_distances(feats):

    distances = cdist(feats, feats, 'cosine')   #cdist - Compute distance between each pair of the two collections of inputs.
    return distances  				#Computes the cosine distance between vectors u and v
    								# i.e dot product of feats and feats


def input_question(data, feats):
    
    if user_response is not None:
        data.insert(0, user_response)
    new_feats = indicoio.text_features(user_response)
    feats.insert(0, new_feats)
    return data, feats


def similarity_text(idx, distance_matrix, data, n_similar=n_similar_val):
  
    t = Texttable()
    t.set_cols_width([50, 20])
    sorted_distance_idxs = np.argsort(distance_matrix[idx])[:n_similar] # EX: [252, 102, 239, ...]
    					#argsort() - Returns the indices that would sort an array.

    					# x = numpy.array([1.48,1.41,0.0,0.1])
						# print x.argsort()

						# >[2 3 1 0]
    most_sim_idx = sorted_distance_idxs[1]
    t.add_rows([['Text', 'Similarity']])
    faq_match = None

    for similar_idx in sorted_distance_idxs:
        datum = data[similar_idx]
        distance = distance_matrix[idx][similar_idx]
        similarity =  1 - distance
        t.add_rows([[datum, str(round(similarity, 2))]])
        if similar_idx == most_sim_idx and similarity >= 0.75:
            faq_match = data[most_sim_idx]
        else:
            sorry = "Sorry, I'm not sure how to respond. Let me find someone who can help you."
    if faq_match is not None:
        fqa={}
        fqa["API_INPUT"]=d["API_INPUT"]
        fqa["MESSAGE"]=faqs[faq_match]
        response=json.dumps(fqa)
        return response

        
    else:
        return sorry
 
def run():
    data = list(faqs.keys())
    with open('C:/Users/krish/Desktop/Data Science/b3ds/speech/faq_voice_bot/faq_feats.pk1', 'rb') as f:
            feats = cPickle.load(f)
   
    input_results = input_question(data, feats)
    new_data = input_results[0]
    new_feats = input_results[1]
    distance_matrix = calculate_distances(new_feats)
    idx = 0
    Faq_response=similarity_text(idx, distance_matrix, new_data)
    return Faq_response

# for recording the questions only, as a user input:
def voice():
    r = sr.Recognizer()
    file = sr.AudioFile('faq_rec4.wav')
    with file as source:
        r.adjust_for_ambient_noise(source, duration = 0.5)          # for removing noise
        audio = r.record(source)

    txt = r.recognize_google(audio)
    text = repr(txt)
    return text

# for recording every questions and answer in a single voice file
# NOTE - Questions must ended with ?..      [PRONOUNCED AS- 'QUESTIONMARK POINT POINT']
#        and Answers must ended with .!     [PRONOUNCED AS - 'POINT EXCLAMATION MARK']
def recording():
    r = sr.Recognizer()
    file = sr.AudioFile('qa_full_recording.wav')
    with file as source:
        r.adjust_for_ambient_noise(source, duration = 0.5)          # for removing noise
        
        audio = r.record(source, duration = 150) #remove the 'duration' argument to transcribe complete audio file
						 #for instance we are here transcribing only first 150 sec i.e. 2.30 minutes

    txt = r.recognize_google(audio)
    text = repr(txt)
    return text

faq_record = recording()

speech = voice()

# for saving the questions and answer in a text file:
with open('transcribed_faq.txt','w') as f:
    f.write(faq_record)

f=open('C:/Users/krish/Desktop/Data Science/b3ds/speech/faq_voice_bot/transcribed_faq.txt','r',errors = 'ignore')
raw=f.read()
raw=raw.lower()


sent_tokens = nltk.sent_tokenize(raw) #sentences can be split into each sentence using sent_tokenize()
word_tokens = nltk.word_tokenize(raw) #A sentence or data can be split into words using the method word_tokenize():
lemmer = nltk.stem.WordNetLemmatizer()
# print(sent_tokens)
#STEMMING- is the process of reducing a word to its word stem that affixes to suffixes and prefixes or to the roots of words known as a lemma.

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(sent_tokens)


def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

    #Lemmatisation (or lemmatization) in linguistics is the process of grouping together the inflected forms of a word so they can be analysed as a single item, identified by the word's lemma, or dictionary form.
    

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation) #The ord() function returns the number representing the unicode code of a specified character.
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

            #normalization that mainly involves eliminating punctuation, converting the entire text into lowercase or uppercase, converting numbers into words, expanding abbreviations, canonicalization of text, and so on.
            #Normalization is a process that converts a list of words to a more uniform sequence. This is useful in preparing text for later processing

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

# Checking for greetings
def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            temp=random.choice(GREETING_RESPONSES)
            V={}
            V["API_INPUT"]=d["API_INPUT"]
            V["MESSAGE"]=temp
            robo_response=json.dumps(V)
            
            
            return robo_response

l0 = []                # for storing the different sentences of a particular cluster
l1 = []
l2 = []

# Generating response
def response(user_response):
    robo_response=''
    # sent_tokens.append(user_response)
    if predicted_cluster == 0:
        l0.append(user_response)
    elif predicted_cluster == 1:
        l1.append(user_response)
    else:
        l2.append(user_response)
    
    FAQ_OUT=run()
    if(FAQ_OUT != "Sorry, I'm not sure how to respond. Let me find someone who can help you."):
        return FAQ_OUT
    else:
        TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
        if predicted_cluster == 0:
                tfidf = TfidfVec.fit_transform(l0)
                vals = cosine_similarity(tfidf[-1], tfidf)            #[-1] is for last element
                idx=vals.argsort()[0][-2]

                flat = vals.flatten()
                flat.sort()
                req_tfidf = flat[-2]
                if(req_tfidf==0):
                    robo_response=robo_response+"I am sorry! I don't understand you"
                    P={}
                    P["API_INPUT"]=d["API_INPUT"]
                    P["MESSAGE"]=robo_response
                    robo_response=json.dumps(P)
                    return robo_response
                else:
                    robo_response = robo_response+l0[idx]

                    Pr={}
                    Pr["API_INPUT"]=d["API_INPUT"]
                    Pr["MESSAGE"]=robo_response
                    robo_response=json.dumps(Pr)
                    # print('robo_response:', robo_response)

                    # robo_response.split(':')
                    robo_response = Pr
                    # print(robo_response)

                    Faq_add=d["MESSAGE"]+":"+robo_response['MESSAGE']
                    print(Faq_add)
                    # print(d['MESSAGE'])
                    y=open('C:/Users/krish/Desktop/Data Science/b3ds/speech/faq_voice_bot/FAQ_UPD.txt','w')
                    y.write(Faq_add)
                    y.close()

                    Pr={}
                    Pr["API_INPUT"]=d["API_INPUT"]
                    Pr["MESSAGE"]=robo_response
                    robo_response=json.dumps(Pr)
                    return robo_response
        elif predicted_cluster == 1:
                        # robo_response = robo_response+l1[idx]
                tfidf = TfidfVec.fit_transform(l1)
                vals = cosine_similarity(tfidf[-1], tfidf)            #[-1] is for last element
                # idx=vals.argsort()[0][-2]
                idx=vals.argsort()[0][-2]
                # print(idx)

                flat = vals.flatten()
                flat.sort()
                req_tfidf = flat[-2]
                if(req_tfidf==0):
                    robo_response=robo_response+"I am sorry! I don't understand you"
                    P={}
                    P["API_INPUT"]=d["API_INPUT"]
                    P["MESSAGE"]=robo_response
                    robo_response=json.dumps(P)
                    return robo_response
                else:
                    robo_response = robo_response+l1[idx]

                    Pr={}
                    Pr["API_INPUT"]=d["API_INPUT"]
                    Pr["MESSAGE"]=robo_response
                    robo_response=json.dumps(Pr)
                    # print('robo_response:', robo_response)

                    # robo_response.split(':')
                    robo_response = Pr
                    # print(robo_response)

                    Faq_add=d["MESSAGE"]+":"+robo_response['MESSAGE']
                    print(Faq_add)
                    # print(d['MESSAGE'])
                    y=open('C:/Users/krish/Desktop/Data Science/b3ds/speech/faq_voice_bot/FAQ_UPD.txt','w')
                    y.write(Faq_add)
                    y.close()
                    Pr={}
                    Pr["API_INPUT"]=d["API_INPUT"]
                    Pr["MESSAGE"]=robo_response
                    robo_response=json.dumps(Pr)
                    return robo_response
        else:
                tfidf = TfidfVec.fit_transform(l2)
                vals = cosine_similarity(tfidf[-1], tfidf)            #[-1] is for last element
                idx=vals.argsort()[0][-2]

                flat = vals.flatten()
                flat.sort()
                req_tfidf = flat[-2]
                if(req_tfidf==0):
                    robo_response=robo_response+"I am sorry! I don't understand you"
                    P={}
                    P["API_INPUT"]=d["API_INPUT"]
                    P["MESSAGE"]=robo_response
                    robo_response=json.dumps(P)
                    return robo_response
                else:
                    robo_response = robo_response+l2[idx]

                    Pr={}
                    Pr["API_INPUT"]=d["API_INPUT"]
                    Pr["MESSAGE"]=robo_response
                    robo_response=json.dumps(Pr)
                    
                    robo_response = Pr
                    # print(robo_response)

                    Faq_add=d["MESSAGE"]+":"+robo_response['MESSAGE']
                    print(Faq_add)
                    # print(d['MESSAGE'])
                    y=open('C:/Users/krish/Desktop/Data Science/b3ds/speech/faq_voice_bot/FAQ_UPD.txt','w')
                    y.write(Faq_add)
                    y.close()
                    Pr={}
                    Pr["API_INPUT"]=d["API_INPUT"]
                    Pr["MESSAGE"]=robo_response
                    robo_response=json.dumps(Pr)
                    return robo_response

        
# used words (axis in our multi-dimensional space)
words = vectorizer.get_feature_names()
n_clusters=3

# for creating clusters using K-Means clustering
def cluster():

    number_of_seeds_to_try=10
    max_iter = 300
    number_of_process=2 # seads are distributed
    model = KMeans(n_clusters=n_clusters, max_iter=max_iter, n_init=number_of_seeds_to_try, n_jobs=number_of_process).fit(X)

    labels = model.labels_
    # indices of preferible words in each cluster
    ordered_words = model.cluster_centers_.argsort()[:, ::-1]

    # print("centers:", model.cluster_centers_)
    print("labels", labels)
    # print("intertia:", model.inertia_)

    labels  = labels.tolist() #for converting the series into list

    get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y] # for getting the indices of a particular cluster 

    label0 = get_indexes(0,labels)    # for storing indices of clusters
    label1 = get_indexes(1,labels)
    label2 = get_indexes(2,labels)

    print('indexes of cluster 0: ',label0)
    print('indexes of cluster 1: ',label1)
    print('indexes of cluster 2: ',label2)

    for i_cluster in range(n_clusters):
        for label in range(n_clusters):
            if label==i_cluster:
                if label == i_cluster == 0:
                    [l0.append(sent_tokens[i]) for i in label0]
                elif label == i_cluster == 1:
                    [l1.append(sent_tokens[i]) for i in label1]
                else:
                    [l2.append(sent_tokens[i]) for i in label2]



    print("\n")
    print("TEXT IN EACH CLUSTER:")
    print()

    print('cluster 0',l0)
    print()
    print('cluster 1',l1)
    print()
    print('cluster 2',l2)
    print()


#for printing top 10 words of each cluster-
    print("Top words per cluster:")
    for i_cluster in range(n_clusters):
        print("Cluster:", i_cluster)
        for term in ordered_words[i_cluster, :10]:
            print("\t"+words[term])

    return model


model1 = cluster()

# using Regular Expressions to filter the answers only
pattern = re.compile(r'question.. [a-zA-Z0-9 .",]+')
pattern2 = re.compile(r' [a-zA-Z0-9 .",]+')

flag=True
print("ROBO: My name is Robo. I will answer your queries about Chatbots. If you want to exit, type Bye!")

while(flag==True):
    # user_input = input('You:')
    user_input = {"API_INPUT":"MESSENGER","MESSAGE":speech}
    		#{"API_INPUT":"MESSENGER","MESSAGE":"When was football played in India?"}
    		# provide the input in the json format
    user_input = json.dumps(user_input)
    d = json.loads(user_input)
    # d = json.loads(voice())
    user_response=d["MESSAGE"]             #user_response is the question asked by A user, like - what is chatbot?

    Y = vectorizer.transform([user_response])
    predicted_cluster = model1.predict(Y)[0]
    
    print('user_response:',user_response)
    print('PREDICTION-')
    print("Cluster:", predicted_cluster)


    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("ROBO: You are welcome..")
        else:
            if(greeting(user_response)!=None):
                print("ROBO: "+greeting(user_response))
            else:
                
                result = pattern.search(response(user_response))

                answer = pattern2.search(result.group())
                print('ROBO: ',answer.group())

                if predicted_cluster == 0:
                    l0.remove(user_response)
                elif predicted_cluster == 1:
                    l1.remove(user_response)
                else:
                    l2.remove(user_response)

                flag = False
    else:
        flag=False
        print("ROBO: Bye! take care..")
