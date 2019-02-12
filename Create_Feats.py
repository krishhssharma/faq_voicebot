import math
import os
from random import sample
import _pickle as cPickle
from tqdm import *

import indicoio
indicoio.config.api_key = "b1b35f65d921fd2d6d0f1933e3f9e99a"

faqs={}
with open('C:/Users/krish/Desktop/Data Science/b3ds/cluster_bot/ipfile.txt') as f:
    for line in f:
        ln=line.split(':')
        n_similar_val=len(ln)+1
        x=ln[0]
        y=ln[1]
        z=len(y)-1
        y=y[0:z]
        faqs[(x)]=y


def make_feats(data):
    """
    Send our text data through the indico API and return each text example's text vector representation
    """
    chunks = [data[x:x+100] for x in range(0, len(data), 100)]
    feats = []
    # just a progress bar to show us how much we have left
    for chunk in tqdm(chunks):
        feats.extend(indicoio.text_features(chunk))    #text_features- Convert text into meaningful feature vectors.
                                                        #Extracts abstract text features for use as inputs to learning algorithms.
    return feats
def run():
    data = list(faqs.keys())
    print ("FAQ data received. Finding features.")
    feats = make_feats(data)
    print(feats)
    with open('faq_feats.pk1', 'wb') as f:
        cPickle.dump(feats, f)
    print ("FAQ features found!")

run()
