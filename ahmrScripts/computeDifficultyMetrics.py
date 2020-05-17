"""
Comput the sentence length and word rarity metrics for aNLI train set from the paper
Save the annotated version
https://arxiv.org/pdf/1903.09848.pdf
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import math
import logging
import transformers

from bert_score import BERTScorer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

# Helper scripts
def genPlots(data):
    # TODO move creating the plots into a separate function
    print("Generating analysis")
    # generate a histogram and cdf
    data = np.asarray(data)
    data_sorted = np.sort(data)
    proportional = np.linspace(0,1,len(data),endpoint=False)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.hist(data,bins=70)
    ax2 = fig.add_subplot(122)
    ax2.plot(data_sorted,proportional)
    plt.savefig("./hist.png",bbox_inches='tight')
    return data_sorted,proportional

def getLine(line):
    js = json.loads(line)
    seq = f"{js['obs1']} {js['obs2']} {js['hyp1']} {js['hyp2']}"
    return seq



"""
Here, the difficulty is based on the total number of words in the example.
The longest example is thus considered to be the hardest.
A new key 'sent_distance' is created and saved into a new dataset
Note that we strip out tokenized periods, so they don't affect the length
"""
def sentenceLength():
    dstFile = open("./sentenceLengthTrain.jsonl", "w+")
    data = []
    # compute the sentence length and append to new dataset
    # total length is defined as obs1 + obs2 + hyp1 + hyp2
    with open("./train.jsonl","r") as srcFile:
        print("Getting seq lengths")
        for line in srcFile:
            seq = getLine(line)
            # tokensize and get rid of periods
            tokens = word_tokenize(seq)
            tokens = list(filter((".").__ne__,tokens))
            l = len(tokens)
            data.append(l)
        line = None
    
    data_sorted,proportional = genPlots(data)
    print("Writing Back Results")
    with open("./train.jsonl","r") as srcFile:

        for i,line in enumerate(srcFile):
            js = json.loads(line)
            # np interp is a liitle lossy, ie the hardest problem will have cdf very close to 1 but not quite
            # so that the data is a true probability distribution you should manually set the easiest and hardest problems to 0.0 and 1.0 respectively
            js["sent_length"] = np.interp(data[i],data_sorted,proportional)
            dstFile.writelines(json.dumps(js))
            dstFile.write('\n')
        
    dstFile.close()
    print("All Done")



"""
Here rare tokens which occurr infrequently in the dataset are thought to make the example harder
This is based on log-probablities calculated from a BOW model of the dataset

Parameters:
lowercase: If True, will convert all text to lowercase before building model
stop_words: If True, removes all words in the NLTK stopword list. The difficulty of a sentnece will then be based
only on the non stop words
lemmatize: If true, uses the lemmatized version of the words to build the model
"""
def wordRarity(lowercase=False):
    dstFile = open("./words.txt", "w+")
    totalWords = 0
    with open("./train.jsonl","r") as srcFile:
        for line in srcFile:
            seq = getLine(line)
            dstFile.writelines(seq.encode("ascii",errors="ignore"))
            dstFile.write("\n")

        dstFile.close()
        dstFile = open("./words.txt", "r")
        # TODO add option for stopwords
        BoW = CountVectorizer(lowercase=lowercase)
        X = BoW.fit_transform(dstFile)
        totalWords = sum(list(BoW.vocabulary_.values()))
        
    
    # Now compute difficulties
    line = None
    dataFile = open("./wordRarityTrain.jsonl","w+")
    data = []   # for generating the graphs
    with open("./train.jsonl","r") as srcFile:
        for line in srcFile:
            seq = getLine(line)
            tokens = word_tokenize(seq)
            tokens = list(filter((".").__ne__,tokens))
            logProb = 0.0
            for token in tokens:
                if BoW.vocabulary_.get(token) == None:
                    # word which have no probability should be skipped
                    continue
                else:
                    logProb += math.log((BoW.vocabulary_.get(token) / totalWords),10)
            data.append(-logProb)

        line = None
        data_sorted,proportional = genPlots(data)
        srcFile = open("./train.jsonl","r")
        for i,line in enumerate(srcFile):
            js = json.loads(line)
            js["rarity"] = np.interp(data[i],data_sorted,proportional)
            dataFile.writelines(json.dumps(js))
            dataFile.write("\n")
        
    
    dataFile.close()



"""
This approach instead looks at the semantic similarity between h1 and h2 using BertScore
If BertScore judge them to be similar, the problem is scored harder because it will presumably
be more difficult to judge which hypothesis is correct
"""
def bertScore():
    with open("./train.jsonl","r") as srcFile:
        all_F1 = []
        all_P = []
        all_R = []
        scorer = BERTScorer(lang='en',rescale_with_baseline=True)
        for line in srcFile:
            js = json.loads(line)
            # the api expects a list of strings
            hyp1 = []
            hyp2 = []
            hyp1.append(js['hyp1'])
            hyp2.append(js['hyp2'])          
            P, R, F1 = scorer.score(hyp1, hyp2, verbose=True)
            all_F1.append(F1)
            all_P.append(P)
            all_R.append(R)
        
        print("here")




            



# Uncomment the one you want
# sentenceLength()
# wordRarity()
bertScore()