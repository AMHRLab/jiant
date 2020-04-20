import nltk
import glob
import unicodedata
import re
import itertools
import json
import random
import datetime
from bs4 import BeautifulSoup



# extract the text from html
# returns the html paragraphs
def ParseHtml():
    allFiles = glob.glob("./gutenberg/*.html")
    docs = {}
    for f in allFiles:
        print(f)
        souped = BeautifulSoup(open(f, 'rb'), features='lxml')
        # get rid of scripts and links, and other junk
        for script in souped(["script","style","link","a","span","br","img","meta"]):
            script.extract()
      
        p = souped.find_all('p')
        docs[f] = p
    
    ExtractParagraphs(docs)

# given the extracted paragraphs, convert to strings and clean
# return the tokenized paragraphs
def ExtractParagraphs(docs):
    # build regexes
    pStart = re.compile(r"<.*>")
    space = re.compile(r"\s{2,1000}")
    for title,paragraphs in docs.items():
        # first get rid of unicode
        # paragraphs = [p.decode(encoding='ascii', errors='ignore') for p in paragraphs]
        # replace various junk
        paragraphs = [str(p).replace('<p>','').replace('</p>','').replace('\n','').replace('\r','').replace('  ',' ').replace('\\','') for p in paragraphs]
        paragraphs = [re.sub(pStart, '', p) for p in paragraphs]   # using regex to catch corner cases
        paragraphs = [re.sub(space, ' ', p) for p in paragraphs]
        paragraphs = [p.strip().replace('“','').replace('”','') for p in paragraphs]
        # remove blank strings
        paragraphs = list(filter(lambda a: a != '', paragraphs))
        paragraphs = list(filter(lambda a: a != ' ', paragraphs))

        # tokenize the paragraphs into sentences
        paragraphs = [nltk.tokenize.sent_tokenize(p) for p in paragraphs]
        # save the result
        docs[title] = paragraphs
    
    BuildEasyDataset(docs)


"""
    Build the easy version of the dataset.
    1 if the sentences are in the same paragraph
    0 if they are in the same story but different paragraphs
"""
def BuildEasyDataset(docs):
    print("Creating easy dataset.")
    dataFile = open("./distanceEasy.jsonl","w+")
    totalPositive = 0 # use so we create an equal number of negative examples
    # the positive examples
    for title,paragraphs in docs.items():
        for i,p in enumerate(paragraphs):
            if len(p) > 1:
                # use itertools to fetch all combinations of the sentences
                allCombs = list(itertools.combinations(p,2))
                for c in allCombs:
                    # build the json example for each
                    # index_0 = p.index(c[0])
                    # index_1 = p.index(c[1])
                    js = {"doc":title,"sentence_1":c[0],"sentence_2":c[1],"label":1}
                    # save and update the positive count
                    dataFile.writelines(json.dumps(js))
                    dataFile.write('\n')
                    totalPositive += 1

    print("Finished creating positive examples")
    
    # the negative examples
    totalNegative = totalPositive
    random.seed(datetime.datetime.now())
    for title,paragraphs in docs.items():
        for i,p in enumerate(paragraphs):
            # exploit the fact that we didn't use single sentence paragraphs to create positive examples
            if len(p) == 1:
                # go through the paragraphs again, but using a random step
                st = random.randint(1,5)
                for ij in range(0,len(paragraphs)-2,st):
                    if ij == i:
                        continue
                    #pick a random sentence in the paragraph
                    sent2 = paragraphs[ij][random.randint(0,len(paragraphs[ij])-1)]
                    js = {"doc":title,"sentence_1":p[0],"sentence_2":sent2,"label":0}
                    dataFile.writelines(json.dumps(js))
                    dataFile.write('\n')
                    totalNegative += 1
                    # stop when the dataset is balanced
                    # TODO: fix so that it actually stops
                    if totalPositive == totalNegative:
                        print("Finished creating negative examples")
                        break



    print("Dataset written")
    dataFile.close()
        


# get the tokenizer
nltk.download('punkt')
ParseHtml()