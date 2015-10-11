__author__ = 'spacegoing'
##
import pickle
from ModelUtils.measures import getAllMeasures, getMeasureCombo, comp_F1_Score
from sklearn.metrics import confusion_matrix
from itertools import combinations
from ModelUtils.trainModel import getDocIndexScoreInfo, getTrainSet
from DataMisc.contestParameters import loadTrainLabels
from sklearn.naive_bayes import GaussianNB


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


def genTrainTestSet(featureMatrix, labels):
    trainFeatureMatrix = featureMatrix
    testFeatureMatrix = featureMatrix
    trainLabels = labels
    testLabels = labels
    return trainFeatureMatrix, trainLabels, testFeatureMatrix, testLabels


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
    modelTrain(trainFeatureMatrix, trainLabels)
    pred = modelPred(testFeatureMatrix)

    return pred

if __name__=="__main__":
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
for measureCombo in measureCombos:
    docIndexScoreInfo = getDocIndexScoreInfo(docIndexLangTrans, docIndexString_Lemma, measureCombo)
    featureMatrix, labels = getTrainSet(docIndexScoreInfo, rawLabels)
    trainFeatureMatrix, trainLabels, \
    testFeatureMatrix, testLabels = genTrainTestSet(featureMatrix, labels)
    pred = runModel(trainFeatureMatrix, trainLabels, testFeatureMatrix, modelClass)
    [[tp, fp], [fn, tn]] = confusion_matrix(testLabels, pred, labelsOrder)



##
