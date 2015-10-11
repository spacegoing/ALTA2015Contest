# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 19:19:26 2015

@author: spacegoing
"""
from DocUtils.concaTextData import readLabels


def returnParas():
    paraFetchTrainTranslations={
    'srcFolder' : "/Users/spacegoing/AllSymlinks/Document Analysis/sharedTask/Data/",
    'fileName' : "Train.txt",
    'keyset' : ['0b8bb0c1-7e51-41f6-8d32-31d15a9ca7ad',\
    '29738119-195a-42c8-ace1-8a78de74b891',\
    'cee2b313-c0c4-46d1-a8b3-a7c60ce254f0']
    }
    
    paraFetchTestTranslations={
    'srcFolder' : "/Users/spacegoing/AllSymlinks/Document Analysis/sharedTask/Data/",
    'fileName' : "Test.txt",
    'keyset' : ['0b8bb0c1-7e51-41f6-8d32-31d15a9ca7ad',\
    '29738119-195a-42c8-ace1-8a78de74b891']
    }
    
    return paraFetchTrainTranslations,  paraFetchTestTranslations


def loadTrainLabels():
    filePath = "/Users/spacegoing/百度云同步盘/macANU/2cdSemester 2015/" \
               "Document Analysis/sharedTask/Code/pycharmVersion/" \
               "Data/Train/Train.csv"
    docLabels = readLabels(filePath)
    return docLabels
