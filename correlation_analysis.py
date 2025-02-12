import copy
from datasets import load_dataset
import csv
import re
import pandas as pd
import krippendorff
from scipy.stats.stats import pearsonr
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

def scatter(x,y,filename,xname,yname):
    plt.scatter(x,y)
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.savefig(filename) 

df = pd.read_csv("all_groups_kripp.csv")
df = df.reset_index()

x = []
y = []
itemDict = get_annotations("Disagree","team_performance")
for index, row in df.iterrows():
    # y.append(row["Agreement"])
    y.append(itemDict[row['Group']][1])
    x.append(itemDict[row['Group']][0])
# scatter(x,y,"Disagree_perform_corr.png","Disagreements","Performance")
print(pearsonr(x,y))




