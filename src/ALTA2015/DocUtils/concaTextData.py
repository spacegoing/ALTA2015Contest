# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 18:06:03 2015

@author: spacegoing
"""
##
import numpy as np
import csv

####
def readALTA2015Data(filepath):
    # File Read lines
    with open(filepath) as f:
        tmpList = []
        for l in f.read().splitlines():
            firstSpace = l.find(" ")
            index = l[:firstSpace]
            String = l[firstSpace:].strip()
            tmpList.append([int(index), String])

        indexString = np.asarray(tmpList, np.object)

    # Generate docIndexString dict{"docid": nparray[[index, String] ... ]}
    titleLineNo = []
    docIndexString = dict()
    # * get line number of title
    for i, s in enumerate(indexString[:, 1]):
        if s[:7] == "<docid ":
            titleLineNo.append(i)
    # * split each docid's block of [index, string] into its docid key
    for i, t in enumerate(titleLineNo[:-1]):
        docIndexString[int(indexString[t, 1][7:-1])] = indexString[t + 1: titleLineNo[i + 1]]
    docIndexString[int(indexString[titleLineNo[-1], 1][7:-1])] = indexString[titleLineNo[-1] + 1:]

    # Generate docStringIndices dict{"docid": [String, Indices[[index, start, end], ... ]]}
    def concaString(indexString):
        indexStartEnd = []  # start, end indices after concatenate (in tmpString)
        tmpString = ""
        length = 0
        for i, s in indexString:
            tmpString += " " + s
            l = len(s)
            indexStartEnd.append([i, length, length + l])
            length += l + 1

        return [tmpString.strip(" "), \
                np.asarray(indexStartEnd)]  # stripped the first " "

    docStringIndices = dict()
    for d in docIndexString:
        docStringIndices[d] = concaString(docIndexString[d])

    # Test if source data contains dirty whitespace (whitespace appears in one term)
    for d in docIndexString:
        if len(docIndexString[d][:, 1]) != len(docStringIndices[d][0].split(" ")):
            raise Warning("Error: File No: " + str(d) + " Dirty source data. Source data contains dirty whitespace \
            (whitespace appears in one term)")
    return docIndexString, docStringIndices


def readLabels(filePath):
    """
    Read Alta 2015 CSV file
    :param filePath:
    :return docLabels: dict{docid: [indexes...] ..., 'column_names':['Eval_id', 'Cognates_id']}
    """

    with open(filePath, newline='') as f:
        reader = csv.reader(f)
        docLabels = dict()
        count = 0
        for row in reader:
            if count == 0:
                docLabels['column_names'] = row
                count += 1
            else:
                docLabels[int(row[0])] = [int(i) for i in row[1].split(' ') if i != '']
    return docLabels

def writeLabels(filePath, docIndexPred, column_names):
    docIndexList = list()
    docIndexList.append(column_names)
    for d,indexList in docIndexPred.items():
        docIndexList.append([d," ".join(str(i) for i in indexList)])
    with open(filePath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(docIndexList)
##
if __name__ == "__main__":
    srcFolder = "/Users/spacegoing/AllSymlinks/Document Analysis/sharedTask/Data/"
    fileName = "result.csv"
    filePath = srcFolder + fileName
    docIndexString, docStringIndices = readALTA2015Data(filePath)
    print(docIndexString[24])
    print("########################")
    print(docStringIndices[24])
##

##
