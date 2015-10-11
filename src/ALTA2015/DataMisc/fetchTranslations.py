# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 15:02:25 2015

@author: spacegoing
"""
from DocUtils.concaTextData import readALTA2015Data
from wsdBabelfyNet import queryBabelNetSimpleLemma, queryBabelfySynsetID


def getTrainRawData(srcFolder, fileName, keyset):
    docIndexString, docStringIndices = readALTA2015Data(srcFolder + fileName)

    # Attention: the following would cost website query limits
    docIndexBabelSynsetID, docFilteredIndexString = \
        queryBabelfySynsetID(docIndexString, docStringIndices)

    docIndexLangTrans = queryBabelNetSimpleLemma(docIndexBabelSynsetID, keyset)

    return docFilteredIndexString, docIndexLangTrans

