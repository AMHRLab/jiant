"""
For the sentence distance task, this scrip will randomly flip the setence assignments of the dataset
The examples are still valid after this, and hopefully it will make harder to trivially predict the
label based on the sentence mappings
"""

import json
import random
from datetime import datetime

dstFile = open("distanceEasyv2.jsonl", "w+")
random.seed(datetime.now())
with open("./distanceEasy.jsonl","r") as srcFile:
    for line in srcFile:
        js = json.loads(line)
        choice = random.randint(0,1)
        if choice == 0:
            # print("Leaving as is")
            dstFile.writelines(json.dumps(js))
            dstFile.write("\n")
        elif choice == 1:
            # print("Flip-flopping")
            temp = js["sentence_2"]
            js["sentence_2"] = js["sentence_1"]
            js["sentence_1"] = temp
            dstFile.writelines(json.dumps(js))
            dstFile.write("\n")
