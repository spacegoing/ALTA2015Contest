from __future__ import division
import math
import os
import Levenshtein
from sklearn.metrics import confusion_matrix


def getMeasureCombo():
    return [xxBigramDice, LCSR, commonBigramNumber, basicNED]


def getAllMeasures():
    return [basicMED, basicNED, jaroDistance, jaroWinklerDistance,
            LCPLength, LCPRatio, LCSLength, LCSR, bigramDice,
            commonBigramNumber, commonBigramRatio, trigramDice,
            commonTrigramNumber, commonTrigramRatio, xBigramDice,
            xxBigramDice, commonXBigramNumber, commonXBigramRatio,
            commonLetterNumber, commonLetterRatio
            ]


### Word Similarity Measures ###
# Returns 1 if at least one letter is shared between the two words.
def sharedLetter(form1, form2):
    for char in form1:
        if char in form2:
            return 1.0
    return 0.0


# Given two n-gram lists, creates a single list that contains all common
# ngrams.
def commonNgrams(ngrams1, ngrams2):
    ngrams2 = ngrams2[:]
    ngrams = []

    for ngram in ngrams1:
        if ngram in ngrams2:
            ngrams.append(ngram)
            ngrams2.remove(ngram)

    return ngrams


# Checks if the two wordforms are identical.
def identicalWords(form1, form2):
    return float(form1 == form2) if (len(form1) * len(form2) > 0) else 0.0


# Checks if the two wordforms have an identical prefix that is at least 4
# characters long.
def identicalPrefix(form1, form2):
    return float(LCPLength(form1, form2) > 3)


# Checks if the two wordforms have the same first letter.
def identicalFirstLetter(form1, form2):
    return float(form1[0] == form2[0]) if (len(form1) * len(form2) > 0) else 0.0


# Computes minimum edit distance between the two wordforms. Here, all edit
# operations have a cost of 1.
def basicMED(form1, form2):
    return float(Levenshtein.distance(form1, form2)) if (len(form1) * len(form2) > 0) else 1.0


# Computes normalized minimum edit distance.
def basicNED(form1, form2):
    return basicMED(form1, form2) / longerWordLen(form1, form2) if (len(form1) * len(form2) > 0) else 1.0


# Computes the Jaro distance between the two words.
def jaroDistance(form1, form2):
    return Levenshtein.jaro(form1, form2) if (len(form1) * len(form2) > 0) else 0.0


# Computes the Jaro-Winkler distance between the two words.
def jaroWinklerDistance(form1, form2):
    return Levenshtein.jaro_winkler(form1, form2, 0.1) if (len(form1) * len(form2) > 0) else 0.0


# Computes the length of the longest common prefix of the two wordforms.
def LCPLength(form1, form2):
    return float(len(os.path.commonprefix([form1, form2])))


# Computes the length of the longest common prefix divided by the length of
# the longer word.
def LCPRatio(form1, form2):
    return LCPLength(form1, form2) / longerWordLen(form1, form2) if (len(form1) * len(form2) > 0) else 0.0


# Computes the length of the longest common subsequence of the two
# wordforms.
def LCSLength(form1, form2):
    lengths = [[0 for j in range(len(form2) + 1)] for i in range(len(form1) + 1)]

    for i, char1 in enumerate(form1):
        for j, char2 in enumerate(form2):
            if char1 == char2:
                lengths[i + 1][j + 1] = lengths[i][j] + 1
            else:
                lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])

    return float(lengths[len(form1)][len(form2)])


# Computes the longest common subsequence ratio (Melamed, 1999).
def LCSR(form1, form2):
    return LCSLength(form1, form2) / longerWordLen(form1, form2) if (len(form1) * len(form2) > 0) else 0.0


# Computes Dice's coefficient based on shared bigrams.
def bigramDice(form1, form2):
    return ngramDice(2, form1, form2)


# Computes Dice's coefficient based on shared trigrams.
def trigramDice(form1, form2):
    return ngramDice(3, form1, form2)


# A variant of Dice's coefficient based on shared extended bigrams (Brew &
# McKelvie, 1996).
def xBigramDice(form1, form2):
    if len(form1) < 3 or len(form2) < 3:
        return 0.0
    else:
        return 2 * commonXBigramNumber(form1, form2) / (len(form1) + len(form2) - 4)


# A variant of Dice's coefficient based on shared extended bigrams and the
# distance between positions of each shared exteded bigram in the two
# wordforms. Each shared bigram thus contributes not 2 to the numerator, but
# rather 2 / (1 + (pos(x) + pos(y))^2), where x and y are the two wordforms
# (Brew & McKelvie, 1996).
def xxBigramDice(form1, form2):
    if len(form1) < 3 or len(form2) < 3:
        return 0.0
    else:
        positions1, positions2 = commonNgramPositions(xBigrams(form1), xBigrams(form2))

        weights = 0.0
        for i, pos1 in enumerate(positions1):
            weights += 2 / (1 + math.pow(pos1 - positions2[i], 2))

        return weights / (len(form1) + len(form2) - 4)


# Computes Dice's coefficient based on n-grams of the two wordforms, where
# s = 2z / x + y (s: similarity, z: number of shared n-grams, x: number of
# n-grams in the first word, and y: number of n-grams in the second word.
def ngramDice(n, form1, form2):
    if len(form1) < n or len(form2) < n:
        return 0.0
    else:
        return 2 * commonNgramNumber(n, form1, form2) / (len(form1) + len(form2) - 2 * (n - 1))


# Computes the number of letters the two words share.
def commonLetterNumber(form1, form2):
    return commonNgramNumber(1, form1, form2)


# Computes the number of bigrams the two words share.
def commonBigramNumber(form1, form2):
    return commonNgramNumber(2, form1, form2)


# Computes the number of trigrams the two words share.
def commonTrigramNumber(form1, form2):
    return commonNgramNumber(3, form1, form2)


# Computes the number of extended bigrams the two words share.
def commonXBigramNumber(form1, form2):
    commonXBigrams = commonNgrams(xBigrams(form1), xBigrams(form2))
    return float(len(commonXBigrams))


# Computes the number of n-grams the two words share.
def commonNgramNumber(n, form1, form2):
    commonNgram = commonNgrams(ngrams(n, form1), ngrams(n, form2))
    return float(len(commonNgram))


# Computes the ratio of shared letters of the two words.
def commonLetterRatio(form1, form2):
    return commonNgramRatio(1, form1, form2)


# Computes the ratio of shared bigrams of the two words.
def commonBigramRatio(form1, form2):
    return commonNgramRatio(2, form1, form2)


# Computes the ratio of shared trigrams of the two words.
def commonTrigramRatio(form1, form2):
    return commonNgramRatio(3, form1, form2)


# Computes the ratio of shared extended bigrams of the two words.
def commonXBigramRatio(form1, form2):
    bigramCount = longerWordLen(form1, form2) - 2
    return commonXBigramNumber(form1, form2) / bigramCount if bigramCount > 0 else 0.0


# Computes the pair's shared n-gram ratio by dividing the number of shared
# n-grams of the two wordforms by the number of n-grams in the longer word.
def commonNgramRatio(n, form1, form2):
    ngramCount = longerWordLen(form1, form2) - (n - 1)
    return commonNgramNumber(n, form1, form2) / ngramCount if ngramCount > 0 else 0.0


# Computes the length of the longer of the two words.
def longerWordLen(form1, form2):
    return float(len(form1)) if len(form1) > len(form2) else float(len(form2))


# Computes the length of the shorter of the two words.
def shorterWordLen(form1, form2):
    return float(len(form1)) if len(form1) < len(form2) else float(len(form2))


# Computes the average word length.
def averageWordLen(form1, form2):
    return float((len(form1) + len(form2)) / 2)


# Computes the absolute difference between the lengths of the two words.
def wordLenDifference(form1, form2):
    return float(abs(len(form1) - len(form2)))


# Computes the relative word length difference between the two words.
def wordLenDifferenceRatio(form1, form2):
    return wordLenDifference(form1, form2) / longerWordLen(form1, form2) if (
        len(form1) > 0 or len(form2) > 0) else 0.0


# Generates a list of the word's n-grams.
def ngrams(n, word):
    return [word[i: i + n] for i in range(len(word) - n + 1)]


# Generates a list of extended bigrams (formed by deleting the middle letter
# from a three-letter substring).
def xBigrams(word):
    return [word[i] + word[i + 2] for i in range(len(word) - 2)]


# Finds positions of shared n-grams within the two wordforms. When the same
# n-gram appears multiple times in a word, preference is given to the n-gram
# closer to the beginning of the word.
def commonNgramPositions(ngrams1, ngrams2):
    ngrams = commonNgrams(ngrams1, ngrams2)

    positions1 = []
    positions2 = []

    for ngram in ngrams:
        for i, ngram1 in enumerate(ngrams1):
            if ngram == ngram1 and i not in positions1:
                positions1.append(i)
                break

        for j, ngram2 in enumerate(ngrams2):
            if ngram == ngram2 and j not in positions2:
                positions2.append(j)
                break

    return positions1, positions2


### Baseline Tests ###
# Returns the wordform it
def getWordform(wordform):
    return wordform


# Returns the first letter of the wordform.
def getFirstLetter(wordform):
    return wordform[0]


# Returns None for wordforms shorter than four characters, and the first
# four characters of the wordform otherwise.
def getPrefix(wordform):
    if len(wordform) < 4:
        return None
    else:
        return wordform[: 4]


def comp_F1_Score(yTrue, yPred, labelsOrder=[1, 0]):
    """

    :param yTrue: 1 is cognates, 0 is non cognates
    :param yPred:
    :param labelsOrder: reorder the tp and fn. if [0,1] then tp is non negative.
    :return:
    """
    [[tp, fp], [fn, tn]] = confusion_matrix(yTrue, yPred, labelsOrder)
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return 2 * p * r / (p + r)
