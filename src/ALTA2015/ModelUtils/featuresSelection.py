__author__ = 'spacegoing'
##
import pickle
from ModelUtils.measures import getAllMeasures, getOptMeasureCombo
from sklearn.metrics import confusion_matrix
from itertools import combinations
from ModelUtils.trainModel import getDocIndexScoreInfo, getTrainSet
from DataMisc.contestParameters import loadTrainLabels
from sklearn.naive_bayes import GaussianNB
from random import shuffle
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
        shuffle(lenMeasureCombos[n])

    return lenMeasureCombos


def genCrossValidData(featureMatrix, labels, kfold=5):
    if labels.ndim < 2:
        labels = labels[:, np.newaxis]

    labelFeaturesLabels = dict()
    labelsOrder = list(np.unique(labels))

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
        crossValiIndex = list(range(0, length + 1, step))
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


def compF1Score(tp, fp, fn):
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return 2 * p * r / (p + r)


def runKfoldValidation(kfold, docIndexLangTrans, docIndexString_Lemma,
                       rawLabels, labelsOrder,
                       modelClass, measureCombos):
    measurenamesDataPerformance = dict()
    for measureCombo in measureCombos:
        docIndexScoreInfo = getDocIndexScoreInfo(docIndexLangTrans, docIndexString_Lemma, measureCombo)
        featureMatrix, labels = getTrainSet(docIndexScoreInfo, rawLabels)
        crossValidData = genCrossValidData(featureMatrix, labels, kfold)

        performanceMatrix = list()
        predOriginlabelsMatrix = list()
        if kfold == 1:
            trainFeatureMatrix, trainLabels, kfoldIndex = \
                crossValidData['kfoldFeatures'][0], crossValidData['kfoldLabels'][0], crossValidData['kfoldIndex']
            testFeatureMatrix = trainFeatureMatrix
            pred = runModel(trainFeatureMatrix, trainLabels, testFeatureMatrix, modelClass)
            [[tp, fp], [fn, tn]] = confusion_matrix(trainLabels, pred, labelsOrder)
            f1 = compF1Score(tp, fp, fn)
            performanceMatrix.append([tp, fp, fn, tn, f1])
            predOriginlabelsMatrix.append(
                {
                    'pred': pred,
                    'originLabels': trainLabels
                }
            )

        else:
            for i in range(kfold):
                trainFeatureMatrix, trainLabels, trainOriginIndex, \
                testFeatureMatrix, testLabels, testOriginIndex = concatKfoldData(crossValidData, i)
                pred = runModel(trainFeatureMatrix, trainLabels, testFeatureMatrix, modelClass)
                [[tp, fp], [fn, tn]] = confusion_matrix(testLabels.ravel(), pred, labelsOrder)
                f1 = compF1Score(tp, fp, fn)
                performanceMatrix.append([tp, fp, fn, tn, f1])
                predOriginlabelsMatrix.append(
                    {
                        'pred': pred,
                        'originLabels': trainLabels
                    }
                )

        performanceMatrix.append(np.sum(np.asarray(performanceMatrix, dtype=np.float64),
                                        axis=0) / kfold
                                 )

        if performanceMatrix[-1][-1] >= 0.8:
            print("FaCaiLa: ", measurenames)

        measurenames = [i.__name__ for i in measureCombo]
        measurenames = "__".join(measurenames)

        measurenamesDataPerformance[measurenames] = {
            'crossValidData': crossValidData,
            'performanceMatrix': performanceMatrix,
            'predOriginlabelsMatrix': predOriginlabelsMatrix
        }

    return measurenamesDataPerformance


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
    # allMeasures = getAllMeasures()
    allMeasures = getOptMeasureCombo()
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
    kfold = 3
    measureCombos = lenMeasureCombos[nMeasures[0]]

    measurenamesDataPerformance = runKfoldValidation(kfold, docIndexLangTrans, docIndexString_Lemma,
                                                     rawLabels, labelsOrder,
                                                     modelClass, measureCombos)

##
choosePerf = list()
for m in measurenamesDataPerformance:
    choosePerf.append([m, measurenamesDataPerformance[m]['performanceMatrix'][-1][-1]])
    print(m)
    pprint(measurenamesDataPerformance[m]['performanceMatrix'])
    print("\n")
choosePerf = np.asarray(choosePerf, dtype=np.object)
choosePerfSorted = choosePerf[choosePerf[:, 1].argsort()]
pprint(choosePerfSorted)
##
# TODO: copy 漏掉的 1. 一定是的
# TODO: 法语 1. 训练集中没预测出来,但在标签里的
