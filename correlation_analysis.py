import copy
from datasets import load_dataset
import csv
import re
import pandas as pd
import krippendorff
import matplotlib.pyplot as plt
delidata_corpus = load_dataset("gkaradzhov/DeliData")

def get_annotations(annotation,output):
    groups = list(delidata_corpus.keys())
    itemDict = {}
    for m in delidata_corpus[groups[0]]:
        #if it is an initial, get starting info
        if(m['message_type'] == 'INITIAL'): 
            itemDict[m['group_id']] = [0,0]
            continue
        #if it is a submit, continue
        elif(m['message_type'] == 'SUBMIT'):
            itemDict[m['group_id']][1] = m[output]
            continue
        #if it is a message, get the text
        else:
            if(m['annotation_target'] == annotation):
                itemDict[m['group_id']][0] += 1

    return itemDict

def scatter(x,y,filename):
    plt.scatter(x,y)
    plt.savefig(filename) 


itemDict = get_annotations("Disagree","team_performance")
print(itemDict)




