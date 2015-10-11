# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 19:14:47 2015

@author: spacegoing
"""
##
import numpy as np
from sklearn.naive_bayes import GaussianNB


def getTrainTestData(docIndexLangTrans, docIndexString_Lemma,
                     test_docIndexLangTrans, test_docIndexString_Lemma,
                     trainLabels, measureCombo):
    # The following two variables contain full Informations.
    # may be need to picle.dump()
    docIndexScoreInfo = getDocIndexScoreInfo(docIndexLangTrans, docIndexString_Lemma, measureCombo)
    test_docIndexScoreInfo = getDocIndexScoreInfo(test_docIndexLangTrans, test_docIndexString_Lemma, measureCombo)

    featureMatrix, labels = getTrainSet(docIndexScoreInfo, trainLabels)

    testSet, testSet_docIndex = getTestSet(test_docIndexScoreInfo)

    return featureMatrix, labels, testSet, testSet_docIndex


def getDocIndexScoreInfo(docIndexLangTrans, docIndexString_Lemma, measureCombo):
    """
    Compute similarities using measureCombo.
    :param docIndexLangTrans: lemmatizeFilteredIndexString(docFilteredIndexString,
                                                    docIndexLangTrans)
    :param docIndexString_Lemma: Same to docIndexLangTrans
    :param measureCombo: from measures import getMeasureCombo
    :return docIndexScoreInfo: {docid: {transScores: [[np.object... ]...], optimal:[np.object... ]
                                    origin:'str'}...}

                            origin: original word (English Lemma).
                            transScores: a list of [french word, score1, score2, ...]
                            optimal: [max(score1),min(score2),...]
                                This is the optimal value of all measurements. May be max may be min,
                                depends on measurements. This behavior need to be changed if use
                                different measurements.
    """
    docIndexScoreInfo = dict()
    for docid in docIndexLangTrans:
        indexMeasures = dict()
        for indexLangTrans in docIndexLangTrans[docid]:
            index = indexLangTrans[0]
            originTerm = docIndexString_Lemma[docid][index]
            transList = indexLangTrans[1]['FR']
            if transList:
                transScores = {'transScores': [[t] + [i(originTerm, t) for i in measureCombo] for t in transList]}

                needMaxMeasures = np.asarray(transScores['transScores'], dtype=np.object)[:, 1:-1]
                needMinMeasures = np.asarray(transScores['transScores'], dtype=np.object)[:, -1]
                optimalScores = {'optimal': np.append(np.max(needMaxMeasures, axis=0), np.min(needMinMeasures))}
                indexMeasures[index] = transScores
                indexMeasures[index].update(optimalScores)
                indexMeasures[index].update({'origin': originTerm})
            else:
                indexMeasures[index] = None
        docIndexScoreInfo[docid] = indexMeasures

    return docIndexScoreInfo



def getTrainSet(docIndexScoreInfo, trainLabels):
    """

    :param docIndexScoreInfo: return of getDocIndexScoreInfo()
    :param trainLabels: from contestParameters import loadTrainLabels
    :return:
        All returns fits the requirements of sklearn.dataset.iris.data and .target
        featureMatrix: np.ndarray float 64
        labels: np.ndarray 1d array float64 contains 0 and 1. 0 for non Cognates. 1 for Cognates.
    """
    cognateList = list()
    nonCognateList = list()
    for d, v in docIndexScoreInfo.items():
        for index, info in v.items():
            if info != None:
                if index in trainLabels[d]:
                    cognateList.append(info['optimal'].astype(np.float64))
                else:
                    nonCognateList.append(info['optimal'].astype(np.float64))
    featureMatrix = np.asarray(cognateList + nonCognateList)
    labels = np.asarray([1 for i in range(len(cognateList))] + [0 for i in range(len(nonCognateList))])

    return featureMatrix, labels


def getTestSet(test_docIndexScoreInfo):
    """

    :param test_docIndexScoreInfo: return of getDocIndexScoreInfo()
    :return:
        np.asarray(measureList): is the same as getTrainSet()'s featureMatrix
        np.asarray(docIndex): np.ndarray np.object [[docid, index]...]
    """
    measureList = list()
    docIndex = list()
    for d, v in test_docIndexScoreInfo.items():
        for index, info in v.items():
            if info != None:
                measureList.append(info['optimal'].astype(np.float64))
                docIndex.append([d, index])

    return np.asarray(measureList), np.asarray(docIndex)


def getGaussianPred(featureMatrix, labels, testSet, testSet_docIndex):
    """
    All input arguments are return of getTrainTestData()
    :param featureMatrix:
    :param labels:
    :param testSet:
    :param testSet_docIndex:
    :return docIndexPred: dict{docid: [index1, index2, ...], ...}
                        key is docid
                        value is all cognates' index
    """
    gnb = GaussianNB()
    gnb.fit(featureMatrix, labels)
    # pred = gnb.predict(featureMatrix)
    pred = gnb.predict(testSet)

    docIndexPred = dict()

    for i, p in enumerate(pred):
        if p:
            docid = testSet_docIndex[i, 0]
            index = testSet_docIndex[i, 1]
            if docid in docIndexPred:
                docIndexPred[docid].append(index)
            else:
                docIndexPred[docid] = [index]

    return docIndexPred


##
if __name__ == "__main__":
    ##
    from DataMisc.contestParameters import loadTrainLabels
    import pickle
    from pprint import pprint
    from ModelUtils.measures import getMeasureCombo

    inputpath = "/Users/spacegoing/百度云同步盘/macANU/" \
                "2cdSemester 2015/Document Analysis/sharedTask" \
                "/Code/pycharmVersion/Data/Train/compMeasurements"
    pkl_file = open(inputpath, 'rb')
    docIndexString_Lemma, docIndexLangTrans, \
    test_docIndexString_Lemma, test_docIndexLangTrans = pickle.load(pkl_file)
    pkl_file.close()

    measureCombo = getMeasureCombo()
    trainLabels = loadTrainLabels()
    featureMatrix, labels, testSet, testSet_docIndex = getTrainTestData(
        docIndexLangTrans, docIndexString_Lemma,
        test_docIndexLangTrans, test_docIndexString_Lemma,
        trainLabels, measureCombo)

    docIndexPred = getGaussianPred(featureMatrix, labels, testSet, testSet_docIndex)

    pprint(docIndexPred)

    ##

# filteredIndex = list()
# for d in docIndexString_Lemma:
#     filteredIndex += list(docIndexString_Lemma[d].keys())
#
# labelsIndex = list()
# for d in trainLabels:
#     labelsIndex += list(trainLabels[d])
#
# missingIndex = list()
# for i in labelsIndex:
#     if i not in filteredIndex:
#         missingIndex.append(i)
# #
# # # Need to import original docIndexString
# inputpath = "/Users/spacegoing/百度云同步盘/macANU/" \
#             "2cdSemester 2015/Document Analysis/sharedTask" \
#             "/Code/pycharmVersion/Data/Train/trainBabelfy"
# pkl_file = open(inputpath, 'rb')
# docIndexString, docStringIndices, docIndexBabelSynsetID, \
# docFilteredIndexString = pickle.load(pkl_file)
# pkl_file.close()
#
# allIndexString = dict()
# for d in docIndexString:
#     for i in docIndexString[d]:
#         allIndexString[i[0]] = i[1]
#
# for i in missingIndex:
#     print(allIndexString[i])

##
