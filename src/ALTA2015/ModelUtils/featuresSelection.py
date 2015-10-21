__author__ = 'spacegoing'
##
import pickle
from ModelUtils.measures import getAllMeasures, getMeasureCombo, comp_F1_Score
from sklearn.metrics import confusion_matrix
from itertools import combinations
from ModelUtils.trainModel import getDocIndexScoreInfo, getTrainSet
from DataMisc.contestParameters import loadTrainLabels
from sklearn.naive_bayes import GaussianNB
import numpy as np
from pprint import pprint


def modelTrainPredMethods(modelClass):
    """
    Used by runModel()
    This will return a instance of modelClass's Train and Pred methods.
    If use a different modelClass, this function may need to be modified.
    :param modelClass: Class.
                    The Model's class,
    :return:  modelClass's Train and Pred methods.
    modelTrain, modelPred
    """
    model = modelClass()
    modelTrain = model.fit
    modelPred = model.predict

    return modelTrain, modelPred


def getAllMeasureCombos(allMeasures, nMeasures):
    """

    :param allMeasuers: from ModelUtils.measures import getAllMeasures
    :param nMeasures: Cnm 's m. Select how many measures from allMeasures
                    This can be a list. nMeasures = [3,4,5]
    :return measureCombos: dict{n: [[allMeasures[1],allMeasures[2],allMeasures[3]],...],...}
    """
    lenMeasureCombos = dict()
    for n in nMeasures:
        combis = list(combinations(range(len(allMeasures)), n))
        lenMeasureCombos[n] = [[allMeasures[i] for i in combi] for combi in combis]

    return lenMeasureCombos


def genCrossValidData(featureMatrix, labels, kfold=5):
    if labels.ndim < 2:
        labels = labels[:, np.newaxis]

    labelFeaturesLabels = dict()

    kfoldFeatures = [np.empty([0, featureMatrix.shape[1]], dtype=np.float64) for i in range(kfold)]
    kfoldLabels = [np.empty([0, labels.shape[1]], dtype=np.float64) for i in range(kfold)]
    kfoldIndex = [np.empty([0], dtype=np.int) for i in range(kfold)]
    for l in labelsOrder:
        index = np.where((labels == l) == True)[0]
        length = index.shape[0]

        # Generate crossValiIndex
        # for example, length = 100, kfold = 3, step = 33
        # in[0]: list(range(0,100,33))
        # out[0]: [0, 33, 66, 99] # Notice the length is 4
        # in[1]: crossValiIndex[-1] = length
        # out[1]: [0, 33, 66, 100] # the last value is the length,
        # namely all the samples are included in the last set
        step = length // kfold
        if kfold == 1:
            crossValiIndex = [0, length]
        else:
            crossValiIndex = list(range(0, length, step))
            crossValiIndex[-1] = length

        labelFeaturesLabels[l] = {'featureMatrix': featureMatrix[index, :],
                                  'labels': labels[index, :],
                                  'length': length,
                                  'crossValiIndex': crossValiIndex,
                                  'kfoldIndex': index
                                  }

        for i, start in enumerate(crossValiIndex[:-1]):
            lFeatures = labelFeaturesLabels[l]['featureMatrix']
            lLabels = labelFeaturesLabels[l]['labels']
            kfoldFeatures[i] = np.concatenate(
                (kfoldFeatures[i], lFeatures[start:crossValiIndex[i + 1]]),
                axis=0)
            kfoldLabels[i] = np.concatenate(
                (kfoldLabels[i], lLabels[start:crossValiIndex[i + 1]]),
                axis=0)
            kfoldIndex[i] = np.concatenate(
                (kfoldIndex[i], index[start:crossValiIndex[i + 1]]),
                axis=0)

    crossValidData = {'labelFeaturesLabels': labelFeaturesLabels,
                      'kfoldFeatures': kfoldFeatures,
                      'kfoldLabels': kfoldLabels,
                      'kfoldIndex': kfoldIndex
                      }
    return crossValidData


def concatKfoldData(crossValidData, i):
    kfoldFeatures, kfoldLabels, kfoldIndex = \
        crossValidData['kfoldFeatures'], crossValidData['kfoldLabels'], crossValidData['kfoldIndex']
    trainFoldIndex = set([i])
    testFoldIndex = set(range(len(kfoldFeatures))) - trainFoldIndex

    trainFeatureMatrix = kfoldFeatures[i]
    trainLabels = kfoldLabels[i]
    trainOriginIndex = kfoldIndex[i]
    testFeatureMatrix = np.concatenate([kfoldFeatures[j] for j in testFoldIndex], axis=0)
    testLabels = np.concatenate([kfoldLabels[j] for j in testFoldIndex], axis=0)
    testOriginIndex = np.concatenate([kfoldIndex[j] for j in testFoldIndex])

    return trainFeatureMatrix, trainLabels, trainOriginIndex, \
           testFeatureMatrix, testLabels, testOriginIndex


def runModel(trainFeatureMatrix, trainLabels, testFeatureMatrix, modelClass):
    """
    If use a different modelClass, modelTrainPredMethods(modelClass) may need to be modified.
    :param trainFeatureMatrix:
    :param trainLabels:
    :param testFeatureMatrix:
    :param modelClass:
    :return: pred result
    """

    modelTrain, modelPred = modelTrainPredMethods(modelClass)
    if trainLabels.ndim == 2:
        modelTrain(trainFeatureMatrix, trainLabels.ravel())
    else:
        modelTrain(trainFeatureMatrix, trainLabels)
    pred = modelPred(testFeatureMatrix)

    return pred


##
if __name__ == "__main__":
    ##
    allMeasures = getAllMeasures()
    nMeasures = [3, 4, 5]
    lenMeasureCombos = getAllMeasureCombos(allMeasures, nMeasures)

    inputpath = "/Users/spacegoing/百度云同步盘/macANU/" \
                "2cdSemester 2015/Document Analysis/sharedTask" \
                "/Code/pycharmVersion/Data/Train/compMeasurements"
    pkl_file = open(inputpath, 'rb')
    docIndexString_Lemma, docIndexLangTrans, \
    test_docIndexString_Lemma, test_docIndexLangTrans = pickle.load(pkl_file)
    pkl_file.close()
    rawLabels = loadTrainLabels()

    labelsOrder = [1, 0]
    modelClass = GaussianNB

# ##
#
# selectedCombo = getMeasureCombo()
# docIndexScoreInfo = getDocIndexScoreInfo(docIndexLangTrans, docIndexString_Lemma, selectedCombo)
# featureMatrix, labels = getTrainSet(docIndexScoreInfo, rawLabels)
# trainFeatureMatrix, trainLabels, \
# testFeatureMatrix, testLabels = genTrainTestSet(featureMatrix, labels)
# pred = runModel(trainFeatureMatrix, trainLabels, testFeatureMatrix, modelClass)
# f1 = comp_F1_Score(testLabels, pred, labelsOrder)
#

##
measureCombos = lenMeasureCombos[nMeasures[0]]
kfold = 1

for measureCombo in measureCombos:
    docIndexScoreInfo = getDocIndexScoreInfo(docIndexLangTrans, docIndexString_Lemma, measureCombo)
    featureMatrix, labels = getTrainSet(docIndexScoreInfo, rawLabels)
    crossValidData = genCrossValidData(featureMatrix, labels, kfold)

    if kfold == 1:
        trainFeatureMatrix, trainLabels, kfoldIndex = \
            crossValidData['kfoldFeatures'][0], crossValidData['kfoldLabels'][0], crossValidData['kfoldIndex']
        testFeatureMatrix = trainFeatureMatrix
        pred = runModel(trainFeatureMatrix, trainLabels, testFeatureMatrix, modelClass)
        [[tp, fp], [fn, tn]] = confusion_matrix(trainLabels, pred, labelsOrder)
    else:
        for i in range(kfold):
            trainFeatureMatrix, trainLabels, trainOriginIndex, \
            testFeatureMatrix, testLabels, testOriginIndex = concatKfoldData(crossValidData, i)
            pred = runModel(trainFeatureMatrix, trainLabels, testFeatureMatrix, modelClass)
            [[tp, fp], [fn, tn]] = confusion_matrix(testLabels.ravel(), pred, labelsOrder)


## conca K fold


##
# TODO: copy 漏掉的
# TODO: 法语
