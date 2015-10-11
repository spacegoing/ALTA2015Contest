# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 18:56:39 2015

@author: spacegoing
"""

##
import numpy as np
import urllib
import json
import gzip


def issueQuery(synsetID, key='cee2b313-c0c4-46d1-a8b3-a7c60ce254f0'):
    service_url = 'https://babelnet.io/v2/getSynset'

    params = {
        'id': synsetID,
        'key': key,
        'filterLangs': ['EN', 'FR']
    }

    url = service_url + '?' + urllib.parse.urlencode(params, 1)
    request = urllib.request.Request(url)
    request.add_header('Accept-encoding', 'gzip')
    response = urllib.request.urlopen(request)

    if response.info().get('Content-Encoding') == 'gzip':
        f = gzip.GzipFile(fileobj=response)
        data = json.loads(f.read().decode())
        return data

    raise Warning("Babelfy no return data, check the website limit")


def getUniqueSynsetIDInfo(docIndexBabelSynsetID, keyset= \
        ['0b8bb0c1-7e51-41f6-8d32-31d15a9ca7ad', \
         '29738119-195a-42c8-ace1-8a78de74b891', \
         'cee2b313-c0c4-46d1-a8b3-a7c60ce254f0'], limit=1000):
    """
    Because the query limit of the webset. This method will first generate a
    unique SynsetID set from docIndexBabelSynsetID. If there are not enough
    keys to query the babelNet, it will raise a warning. 
    
    Parameters
    ----
    docIndexBabelSynsetID : dict(docid:[[index, synsetID]...])
        wsdBabelfyNe.getDocIndexBabelSynsetID's return
    keyset : list[key1, key2]
        Keys of babelNet.com
    limit : int
        babelNet.com's query limit
        
    Return
    ----
    uniqueSynIDBabelNetDict : { "bn:00001422n" : issueQuery("bn:00001422n", key) }
        Key is the bnID of babelNet. Value is issueQuery()'s return.
        Each key is unique. This result is supposed to be the input of method:
        uniqueSynIDDict2docIndexLangSimpleLemma()
        
    Example
    ----
    >>> docIndexBabelSynsetID = getDocIndexBabelSynsetID(docIndexString, docStringIndices)
    >>> uniqueSynIDBabelNetDict = getUniqueSynsetIDInfo(docIndexBabelSynsetID)
    """
    uniqueSynSetID = set()
    for d in docIndexBabelSynsetID:
        for bn in docIndexBabelSynsetID[d][:, 1]:
            uniqueSynSetID.add(bn)
    uniqueSynSetID = np.asarray(list(uniqueSynSetID))

    noOfUnique = uniqueSynSetID.shape[0]
    if len(keyset) * limit < noOfUnique:
        raise Warning("Not enough keys. Exceed website daily limit")
    setIndicesUnderLimit = list(range(0, noOfUnique, 1000))
    setIndicesUnderLimit.append(noOfUnique)

    separatedSynList = list()
    for i, ind in enumerate(setIndicesUnderLimit[:-1]):
        separatedSynList.append(uniqueSynSetID[ind: setIndicesUnderLimit[i + 1]])

    uniqueSynIDBabelNetDict = dict()
    for syn, key in zip(separatedSynList, keyset):
        for s in syn:
            uniqueSynIDBabelNetDict[s] = issueQuery(s, key)

    return uniqueSynIDBabelNetDict


def uniqueSynIDDict2docIndexLangSimpleLemma(uniqueSynIDBabelNetDict, docIndexBabelSynsetID):
    """
    This will assign each uniqueSynIDBabelNetDict item to its belonging doc's
    index's synsetID in docIndexBabelSynsetID. It will return a new dict contains
    docIndexLangTrans.
    
    This will convert utf-8 to byte code (str()) and store the lower case.
    
    Parameters
    ----
    docIndexBabelSynsetID : dict(docid:[[index, synsetID]...])
        wsdBabelfyNe.getDocIndexBabelSynsetID's return
    uniqueSynIDBabelNetDict : { "bn:00001422n" : issueQuery("bn:00001422n", key) }
        return of getUniqueSynsetIDInfo()
        
    Return
    ----
    docIndexLangTrans : {docid: np.array([index, {"EN": [simpleLemma...], "FR": [simpleLemma...]}]...)}
        Dict contains docid, each docs' all indexes and their simple lemmas in EN and FR
        
        
    Example
    ----
    >>> docIndexBabelSynsetID = getDocIndexBabelSynsetID(docIndexString, docStringIndices)
    >>> uniqueSynIDBabelNetDict = getUniqueSynsetIDInfo(docIndexBabelSynsetID)
    >>> docIndexLangTrans = \
    uniqueSynIDDict2docIndexLangSimpleLemma(uniqueSynIDBabelNetDict, docIndexBabelSynsetID)
    """
    synIDLangTranDict = dict()
    for u, v in uniqueSynIDBabelNetDict.items():
        langTranDict = {"EN": set(), "FR": set()}
        for sense in v['senses']:
            langTranDict[sense["language"]].add(sense["simpleLemma"].lower())
        for l in langTranDict:
            langTranDict[l] = list(langTranDict[l])
        synIDLangTranDict[u] = langTranDict

    docIndexLangTrans = dict()
    for d, v in docIndexBabelSynsetID.items():
        indexLangTrans = list()
        for index, synsetID in v:
            indexLangTrans.append([index, synIDLangTranDict[synsetID]])

        docIndexLangTrans[d] = np.asarray(indexLangTrans, np.object)

    return docIndexLangTrans


##
if __name__ == "__main__":
    import pickle

    inputpath = "/Users/spacegoing/百度云同步盘/macANU/" \
                "2cdSemester 2015/Document Analysis/sharedTask" \
                "/Code/pycharmVersion/Data/Train/trainBabelfy"
    pkl_file = open(inputpath, 'rb')

    docIndexString, docStringIndices, docIndexBabelSynsetID, \
    docFilteredIndexString = pickle.load(pkl_file)
    pkl_file.close()

    # # Attention, this will consume website limits
    # uniqueSynIDBabelNetDict = getUniqueSynsetIDInfo(docIndexBabelSynsetID)
    #
    # outputBabelNetPath = "/Users/spacegoing/百度云同步盘/macANU/" \
    #           "2cdSemester 2015/Document Analysis/sharedTask" \
    #           "/Code/pycharmVersion/Data/Test/testBabelNet"
    # outputBabelNet = open(outputBabelNetPath,"wb")
    # pickle.dump(uniqueSynIDBabelNetDict, outputBabelNet, -1)
    # outputBabelNet.close()

    inputBabelNetPath = "/Users/spacegoing/百度云同步盘/macANU/" \
                        "2cdSemester 2015/Document Analysis/sharedTask" \
                        "/Code/pycharmVersion/Data/Test/testBabelNet"
    pkl_file = open(inputBabelNetPath, 'rb')
    uniqueSynIDBabelNetDict = pickle.load(pkl_file)
    pkl_file.close()

    docIndexLangTrans = \
        uniqueSynIDDict2docIndexLangSimpleLemma(uniqueSynIDBabelNetDict, docIndexBabelSynsetID)

    outputdocIndexLangTransPath = "/Users/spacegoing/百度云同步盘/macANU/" \
                                  "2cdSemester 2015/Document Analysis/sharedTask" \
                                  "/Code/pycharmVersion/Data/Test/testBabelNetTotall"
    outputdocIndexLangTrans = open(outputdocIndexLangTransPath, "wb")
    pickle.dump([uniqueSynIDBabelNetDict, docIndexLangTrans], outputdocIndexLangTrans, -1)
    outputdocIndexLangTrans.close()
