# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 16:46:57 2015

@author: spacegoing
"""
from DocUtils.concaTextData import readALTA2015Data
from queryBabelfy import getDocBnSets, getFilteredDocBnSets
from queryBabelNet import getUniqueSynsetIDInfo,\
uniqueSynIDDict2docIndexLangSimpleLemma

def queryBabelfySynsetID(docIndexString, docStringIndices,\
key = '29738119-195a-42c8-ace1-8a78de74b891'):
    """
    Return
    ----
    docIndexBabelSynsetID: dict{docid: np.ndarray([index, synSetID]...)}
        index is the term's index in source data file
        
    Example
    ----
    >>> srcFolder = "/Users/spacegoing/AllSymlinks/Document Analysis/sharedTask/Data/"
    >>> fileName = "Train.txt"
    >>> docIndexString, docStringIndices = readALTA2015Data(srcFolder + fileName)
    >>> docIndexBabelSynsetID, docFilteredIndexString = queryBabelfySynsetID(docIndexString, docStringIndices)
    """
    
    rawDocBnSets = getDocBnSets(docStringIndices)

    docIndexBabelSynsetID, docFilteredIndexString = \
        getFilteredDocBnSets(docStringIndices, rawDocBnSets)
    
    return docIndexBabelSynsetID, docFilteredIndexString

def queryBabelNetSimpleLemma(docIndexBabelSynsetID, keyset =
    ['0b8bb0c1-7e51-41f6-8d32-31d15a9ca7ad',\
    '29738119-195a-42c8-ace1-8a78de74b891'], limit = 1000):
    """
    This will assign each uniqueSynIDBabelNetDict item to its belonging doc's
    index's synsetID in docIndexBabelSynsetID. It will return a new dict contains
    docIndexLangTrans which only contains each translation's simple lemma.
    
    Parameters
    ----
    docIndexBabelSynsetID : dict(docid:[[index, synsetID]...])
        wsdBabelfyNe.getDocIndexBabelSynsetID's return
        
    Return
    ----
    docIndexLangTrans : {docid: np.array([index, {"EN": [simpleLemma...], "FR": [simpleLemma...]}]...)}
        Dict contains docid, each docs' all indexes and their simple lemmas in EN and FR
        
        
    Example
    ----
    >>> docIndexBabelSynsetID = queryBabelfySynsetID(docIndexString, docStringIndices)
    >>> docIndexLangTrans = queryBabelNetSimpleLemma(docIndexBabelSynsetID)
    """
    
    uniqueSynIDBabelNetDict = getUniqueSynsetIDInfo(docIndexBabelSynsetID,\
    keyset, limit)

    return uniqueSynIDDict2docIndexLangSimpleLemma\
    (uniqueSynIDBabelNetDict, docIndexBabelSynsetID)


##
if __name__ == "__main__":
    from configFile import returnParams
    websiteParams, filePath = returnParams()
    docIndexString, docStringIndices = readALTA2015Data(filePath)
    docIndexBabelSynsetID, docFilteredIndexString = queryBabelfySynsetID(docIndexString, \
    docStringIndices, websiteParams['keyset'][0])
    
    keyset = websiteParams['keyset']
    limit = websiteParams['limit']
    docIndexLangTrans = queryBabelNetSimpleLemma(docIndexBabelSynsetID, **websiteParams)
#    # Pickle the synsetID
#    import pickle
#    outputSysetIDSetPath = "/Users/spacegoing/AllSymlinks/Document Analysis/sharedTask/Data/testSynsetID.pkl"
#    outputSysetID = open(outputSysetIDSetPath,"wb")
#    pickle.dump(docIndexBabelSynsetID, outputSysetID, -1)
#    outputSysetID.close()
    
#    outputdocFilteredIndexStringPath = \
#    "/Users/spacegoing/AllSymlinks/Document Analysis/sharedTask/Data/testdocFilteredIndexString.pkl"
#    outputdocFilteredIndexString= open(outputdocFilteredIndexStringPath,"wb")
#    pickle.dump(docFilteredIndexString, outputdocFilteredIndexString, -1)
#    outputdocFilteredIndexString.close()