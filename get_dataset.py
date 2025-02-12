import copy
from datasets import load_dataset
import csv
import re
import pandas as pd
import krippendorff
import matplotlib.pyplot as plt
delidata_corpus = load_dataset("gkaradzhov/DeliData")

def get_features():
    groups = list(delidata_corpus.keys())
    groupDict = {}
    submissionDict = {}
    feature = 'original_text'
    #for every row, check the type
    for m in delidata_corpus[groups[0]]:
        #if it is an initial, get starting info
        if(m['message_type'] == 'INITIAL'): 
            groupDict[m['group_id']] = []
            submissionDict[m['group_id']] = {}
            group_members = m[feature].split("&&")[0].replace("SYSTEM","").replace(",,",",").split(",")
            cards = m[feature].split("&&")[1].split(",")
            submissionDict[m['group_id']]["cards"] = cards
            for member in group_members:
                submissionDict[m['group_id']][member] = []
            continue
        #if it is a submit, get sumbission info
        elif(m['message_type'] == 'SUBMIT'): 
            submissionDict[m['group_id']][m['origin']] = str(m[feature]).split(",")
            continue
        #if it is a message, get the text
        else:
            groupDict[m['group_id']].append(m[feature])
    return groupDict, submissionDict

def get_agreement(submissionDict):
    agreementDict = {}
    for group in submissionDict.keys():
        cards = submissionDict[group]["cards"]
        submissions = []
        # Create a raters x cards array
        for origin in submissionDict[group].keys():
            if (origin == "cards"):continue
            origin_answers = [0]*len(cards)
            for answer in submissionDict[group][origin]:
                #If they submit cards, find the correct index and mark it
                if(answer in cards):
                    origin_answers[cards.index(answer)] = 1
            submissions.append(origin_answers)
        alpha = krippendorff.alpha(reliability_data = submissions,level_of_measurement="nominal",value_domain=[0,1])
        if(alpha <= 0):
            agreementDict[group] = krippendorff.alpha(reliability_data = submissions,level_of_measurement="nominal",value_domain=[0,1])
    return agreementDict

def save_to_csv(filename,header,groupDict,submissionDict,agreementDict):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        if header:
            writer.writerow(header)
        for key in agreementDict.keys():
            writer.writerow([key,groupDict[key],submissionDict[key],agreementDict[key]])

def plot_hist(agreementDict,filename):
    agreement = []
    for key in agreementDict.keys():
        #If they all chose everything or nothing, krip gives a nan; make it a 1
        if(agreementDict[key] != agreementDict[key]): 
            agreement.append(1)
            continue
        agreement.append(agreementDict[key])
    print(sum(agreement)/len(agreement))
    plt.hist(agreement)
    plt.savefig(filename) 

#groupDict is the text, submissionDict is their answers
groupDict, submissionDict = get_features()
#agreementDict is their agreement scores
agreementDict = get_agreement(submissionDict)
plot_hist(agreementDict,'disagreements_hist.png')
save_to_csv("all_negative_kripp.csv",["Group","Text","Submissions","Agreement"],groupDict,submissionDict,agreementDict)



