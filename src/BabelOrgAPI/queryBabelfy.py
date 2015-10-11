# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 22:20:06 2015

@author: spacegoing
"""
##
import numpy as np
import urllib
import json
import gzip


def issueQuery(text, key='29738119-195a-42c8-ace1-8a78de74b891'):
    """

    :param text:
    :param key:
    :return data: list[dict{},...]
    """
    service_url = 'https://babelfy.io/v1/disambiguate'
    lang = 'EN'
    text = "Russia has stepped up its rhetoric against Ukraine's new "

    params = {
        'text': text,
        'lang': lang,
        'key': key,
        'match': 'EXACT_MATCHING',
    }

    url = service_url + '?' + urllib.parse.urlencode(params)
    request = urllib.request.Request(url)
    request.add_header('Accept-encoding', 'gzip')
    response = urllib.request.urlopen(request)

    if response.info().get('Content-Encoding') == 'gzip':
        f = gzip.GzipFile(fileobj=response)
        data = json.loads(f.read().decode())
        return data

    raise Warning("Babelfy no return data, check the website limit")


def getDocBnSets(docStringIndices):
    rawDocBnSets = dict()
    for d in docStringIndices:
        rawDocBnSets[d] = issueQuery(docStringIndices[d][0])

    return rawDocBnSets


def filterRawBnSets(rawBnSets, indices, document):
    """
    Filter multi-terms' senses. Assign babelSynsetID to its index (returned by
    readALTA2015Data() ) accordingly.
    
    Parameters
    ----
    rawBnSets : issueQuery()'s return
        Contains both term sense and multi-terms' sense
    indices : readALTA2015Data()'s second return
        docStringIndices[d][1]. namely document d 's indices np.ndarray([index, start, end]...)
    document : String
        docStringIndices[d][0]. namely document d 's entire content.
    Return
    ----
    filteredBnSets : np.ndarray([index, babelSynsetID]...) np.object
        bnSets filtered. together with its index
    filteredIndexString : np.ndarray([index, filteredString]...) np.object
        Filter raw strings using returned BnSets' charFragment. This will 
        filter string with punctuations like "chocolate," to "chocolate" and
        hopefully will return lemma to some exstance. 
        
    Example
    ----
    >>> docIndexString, docStringIndices = readALTA2015Data(srcFolder + fileName)
    >>> indices = docStringIndices[1][1]
    >>> rawString = docStringIndices[1][0]
    >>> rawBnSets = issueQuery(rawString)
    >>> filterRawBnSets(rawBnSets, indices)
    """
    originIndices = indices[:, 1:]  # np.adarray([start, end]...)
    index = indices[:, 0]  # np.adarray([index]...)
    iterIndex = list(range(index.shape[0]))
    indexBnSets = list()
    filteredStrings = list()
    for r in rawBnSets:
        charFragment = r.get('charFragment')
        cfStart = charFragment.get('start')
        cfEnd = charFragment.get('end')
        for i, ind in enumerate(iterIndex):
            if originIndices[ind, 0] <= cfStart and cfEnd <= originIndices[ind, 1]:
                indexBnSets.append([index[ind], r.get('babelSynsetID')])
                filteredStrings.append([index[ind],
                                        document[cfStart:cfEnd + 1].lower()])
                iterIndex.pop(i)
                break
    filteredBnSets = np.asarray(indexBnSets, np.object)
    filteredStrings = np.asarray(filteredStrings, np.object)
    return filteredBnSets, filteredStrings  # np.ndarray([index, babelSynsetID]...) np.object


def getFilteredDocBnSets(docStringIndices, rawDocBnSets):
    docIndexBabelSynsetID = dict()
    docFilteredIndexString = dict()
    for d in docIndexString:
        rawBnSets = rawDocBnSets[d]

        indices = docStringIndices[d][1]
        document = docStringIndices[d][0]
        filteredBnSets, filteredStrings = filterRawBnSets(rawBnSets, indices, document)

        docIndexBabelSynsetID[d] = filteredBnSets
        docFilteredIndexString[d] = filteredStrings

    return docIndexBabelSynsetID, docFilteredIndexString


##
if __name__ == "__main__":
    from DocUtils.concaTextData import readALTA2015Data

    filepath = "/Users/spacegoing/百度云同步盘/macANU/" \
               "2cdSemester 2015/Document Analysis/sharedTask" \
               "/Code/pycharmVersion/Data/Test/Test.txt"
    docIndexString, docStringIndices = readALTA2015Data(filepath)
    rawDocBnSets = getDocBnSets(docStringIndices)

    # rawBnSets = rawDocBnSets[24]
    # indices = docStringIndices[24][1]
    # indexStrings = docStringIndices[24][0]
    # filteredBnSets, filteredStrings = filterRawBnSets(rawBnSets, indices, indexStrings)
    # print(filteredBnSets[:5])
    # print(filteredBnSets[-5:])

    docIndexBabelSynsetID, docFilteredIndexString = \
        getFilteredDocBnSets(docStringIndices, rawDocBnSets)

    import pickle

    babelfyData = [docIndexString, docStringIndices,
                   docIndexBabelSynsetID, docFilteredIndexString]
    outpath = "/Users/spacegoing/百度云同步盘/macANU/" \
              "2cdSemester 2015/Document Analysis/sharedTask" \
              "/Code/pycharmVersion/Data/Test/testBabelfy"
    outfile = open(outpath, "wb")
    pickle.dump(babelfyData, outfile, -1)
    outfile.close()
