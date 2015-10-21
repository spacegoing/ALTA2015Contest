# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 19:15:39 2015

@author: spacegoing
"""

## Initial Params
from pprint import pprint

from DocUtils.concaTextData import writeLabels
from DataMisc.contestParameters import returnParas, loadTrainLabels
from DataMisc.loadPickleData import loadTrainData, loadTestData
from DocUtils.procCognates import lemmatizeFilteredIndexString
from ModelUtils.measures import getMeasureCombo
from ModelUtils.trainModel import getTrainTestData, getGaussianPred

paraFetchTrainTranslations, paraFetchTestTranslations = returnParas()

# Get TrainData & Test Data
docFilteredIndexString, docIndexBabelSynsetID, docIndexLangTrans = loadTrainData()
test_docFilteredIndexString, test_docIndexBabelSynsetID, test_docIndexLangTrans = loadTestData()

trainLabels = loadTrainLabels()

# Lemmatization Data
docIndexString_Lemma = lemmatizeFilteredIndexString(docFilteredIndexString,
                                                    docIndexLangTrans)
test_docIndexString_Lemma = lemmatizeFilteredIndexString(test_docFilteredIndexString,
                                                         test_docIndexLangTrans)

# import pickle
#
# trainData = [docIndexString_Lemma, docIndexLangTrans,
#              test_docIndexString_Lemma, test_docIndexLangTrans]
# outpath = "/Users/spacegoing/百度云同步盘/macANU/" \
#           "2cdSemester 2015/Document Analysis/sharedTask" \
#           "/Code/pycharmVersion/Data/Train/compMeasurements"
# outfile = open(outpath, "wb")
# pickle.dump(trainData, outfile, -1)
# outfile.close()

# Get Train and Test Data
measureCombo = getMeasureCombo()

featureMatrix, labels, testSet, testSet_docIndex = getTrainTestData(
    docIndexLangTrans, docIndexString_Lemma,
    test_docIndexLangTrans, test_docIndexString_Lemma,
    trainLabels, measureCombo)

# Run classification
docIndexPred = getGaussianPred(featureMatrix, labels, testSet, testSet_docIndex)

pprint(docIndexPred)

filePath = "/Users/spacegoing/百度云同步盘/macANU/2cdSemester 2015/" \
           "Document Analysis/sharedTask/Code/pycharmVersion/" \
           "Data/testResult4.csv"
writeLabels(filePath,docIndexPred,trainLabels['column_names'])
##
