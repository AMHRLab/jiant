"""
Comput the sentence length and word rarity metrics for aNLI train set from the paper
Save the annotated version
https://arxiv.org/pdf/1903.09848.pdf
"""

import json
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import numpy as np

def sentenceLength():
    dstFile = open("./sentenceLengthTrain.jsonl", "w+")
    data = []
    # compute the sentence length and append to new dataset
    # total length is defined as obs1 + obs2 + hyp1 + hyp2
    with open("./train.jsonl","r") as srcFile:
        print("Getting seq lengths")
        for line in srcFile:
            js = json.loads(line)
            seq = f"{js['obs1']} {js['obs2']} {js['hyp1']} {js['hyp2']}"
            # tokensize and get rid of periods
            tokens = word_tokenize(seq)
            tokens = list(filter((".").__ne__,tokens))
            l = len(tokens)
            data.append(l)
        line = None
        
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



    print("Writing Back Results")
    with open("./train.jsonl","r") as srcFile:

        for i,line in enumerate(srcFile):
            js = json.loads(line)
            # np interp is a liitle lossy, ie the hardest problem will have cdf very close to 1 but not quite
            # so that the data is a true probability distribution you should manually set the hardest problem to 1.0
            js["sent_length"] = np.interp(data[i],data_sorted,proportional)
            dstFile.writelines(json.dumps(js))
            dstFile.write('\n')
        
    dstFile.close()
    print("All Done")



# TODO
def wordRarity():
    dstFile = open("./wordRarityTrain", "w+")
    with open("./train.jsonl","r") as srcFile:
        pass


sentenceLength()