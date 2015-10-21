__author__ = 'spacegoing'
##
from nltk.stem import WordNetLemmatizer


# OpenNLP CoreNLP GATE

def show_didnt_match_result(docFilteredIndexString, docIndexLangTrans):
    docValidateData = {k: {i[0]: i[1] for i in v}
                       for k, v in docFilteredIndexString.items()}

    for k, v in docValidateData.items():
        print("###############################\n")
        print("Document: " + str(k))
        for i in docIndexLangTrans[k]:
            if v[i[0]].lower() not in i[1]["EN"]:
                print(v[i[0]] + "\n")
                print(i[1]["EN"])


def show_didnt_match_result_after_lemma(docIndexString_Lemma, docIndexLangTrans):
    for k, v in docIndexString_Lemma.items():
        print("###############################\n")
        print("Document: " + str(k))
        for i in docIndexLangTrans[k]:
            if v[i[0]].lower() not in i[1]["EN"]:
                print("\n" + v[i[0]])
                print(i[1]["EN"])


def lemmatizeFilteredIndexString(docFilteredIndexString, docIndexLangTrans):
    """

    :param docFilteredIndexString:
    :param docIndexLangTrans:
    :return:
    """
    wnl = WordNetLemmatizer()

    docIndexString_Lemma = {k: {i[0]: i[1] for i in v}
                            for k, v in docFilteredIndexString.items()}

    def simpleSimi(originalTerm, candidateTerm):
        score = 0
        s1 = list(originalTerm)
        s2 = list(candidateTerm)
        for i in s1:
            if i == s2[0]:
                score += 1
                s2.pop(0)
            else:
                break
        return score

    for docid, indexStringDict in docIndexString_Lemma.items():
        for indexLangTrans in docIndexLangTrans[docid]:
            originTerm = indexStringDict[indexLangTrans[0]]  # The word in original text
            enTrans = indexLangTrans[1]["EN"]

            if originTerm not in enTrans:  # If originTerm is not lemmatized
                # This part use BabelNet's translations to lemmatize originTerm:
                # If translation e is part of originTerm, then e is it's lemma
                shortest_Term = float("inf")
                for e in enTrans:
                    if e in originTerm and len(e) <= shortest_Term:
                        indexStringDict[indexLangTrans[0]] = e
                        shortest_Term = len(e)

                # This part use WordNet's lemmatizer.
                # The wnl.lemmatize() requires post tag. Otherwise the result will
                # be the same as newOriginTerm.
                #
                # However, the nltk.pos_tag()'s types are not the same as wordnet's type.
                # So we can't get Post Tag. But after the upper part, all adv, adj and so
                # on will be lemmatized. Only the Noun and Verb will be left.
                #
                # As a workaround. This algorithm will first assume the newOriginTerm is
                # noun(default). If it is not, then it will check if it's Verb.
                #
                # If non of this take effect, the algorithm will assign the shortest most
                # similar (measured by simpleSimi()) word in enTrans as the lemma. And it
                # will be printed.
                newOriginTerm = indexStringDict[indexLangTrans[0]]
                if newOriginTerm not in enTrans:
                    nounterm = wnl.lemmatize(newOriginTerm)
                    if nounterm in enTrans:  # Check if it's Noun
                        indexStringDict[indexLangTrans[0]] = nounterm
                    else:
                        verbTerm = wnl.lemmatize(newOriginTerm, 'v')
                        if verbTerm in enTrans:  # Check if it's Verb
                            indexStringDict[indexLangTrans[0]] = verbTerm
                        else:

                            # If non of the upper works, assign it to the shortest most
                            # similar (measured by simpleSimi()) word in enTrans as the lemma.
                            scores = list()
                            for e in enTrans:
                                scores.append(simpleSimi(newOriginTerm, e))
                            mScore = max(scores)
                            mScoreIndices = [i for i, j in enumerate(scores) if j == mScore]

                            shortestLen = float('inf')
                            shortestIndice = 0
                            for m in mScoreIndices:
                                if len(enTrans[m]) <= shortestLen:
                                    shortestLen = len(enTrans[m])
                                    shortestIndice = m

                            fs = enTrans[shortestIndice]
                            indexStringDict[indexLangTrans[0]] = fs

                            print("Undetermined Lemmatization: ")
                            print("Document ID: " + str(docid))
                            print("Origin Term: " + newOriginTerm)
                            print("wordnet: " + verbTerm)
                            print("BabelNet Translations: ")
                            print(enTrans)
                            print("Finally mapped to: " + fs + "\n")

    return docIndexString_Lemma


mapFREN = {'À': 'a',
           'Â': 'a',
           'Ä': 'a',
           'Æ': 'a',
           'Ç': 'c',
           'È': 'e',
           'É': 'e',
           'Ê': 'e',
           'Ë': 'e',
           'Î': 'i',
           'Ï': 'i',
           'Ô': 'o',
           'Ö': 'o',
           'Ù': 'u',
           'Û': 'u',
           'Ü': 'u',
           'à': 'a',
           'â': 'a',
           'ä': 'a',
           'æ': 'a',
           'ç': 'c',
           'è': 'e',
           'é': 'e',
           'ê': 'e',
           'ë': 'e',
           'î': 'i',
           'ï': 'i',
           'ô': 'o',
           'ö': 'o',
           'ù': 'u',
           'û': 'u',
           'ü': 'u',
           'ÿ': 'y',
           'Œ': 'o',
           'œ': 'o',
           'Ÿ': 'y'}


def mapFrenchEnglishCharacters():
    ##

    import pickle
    inputpath = "/Users/spacegoing/百度云同步盘/macANU/" \
                "2cdSemester 2015/Document Analysis/sharedTask" \
                "/Code/pycharmVersion/Data/Train/trainBabelNet"
    pkl_file = open(inputpath, 'rb')
    uniqueSynIDBabelNetDict, docIndexLangTrans = pickle.load(pkl_file)
    pkl_file.close()

    docIndexLangTrans = mapIndexLangTrans(docIndexLangTrans, mapFREN)

    outpath = "/Users/spacegoing/百度云同步盘/macANU/" \
                "2cdSemester 2015/Document Analysis/sharedTask" \
                "/Code/pycharmVersion/Data/Train/trainBabelNet_Mapped"
    pkl_file = open(outpath, 'wb')
    pickle.dump([uniqueSynIDBabelNetDict, docIndexLangTrans],pkl_file,-1)
    pkl_file.close()


    inputpath = "/Users/spacegoing/百度云同步盘/macANU/" \
                "2cdSemester 2015/Document Analysis/sharedTask" \
                "/Code/pycharmVersion/Data/Test/testBabelNetTotall"
    pkl_file = open(inputpath, 'rb')
    uniqueSynIDBabelNetDict, test_docIndexLangTrans = pickle.load(pkl_file)
    pkl_file.close()

    test_docIndexLangTrans = mapIndexLangTrans(test_docIndexLangTrans, mapFREN)

    outpath = "/Users/spacegoing/百度云同步盘/macANU/" \
                "2cdSemester 2015/Document Analysis/sharedTask" \
                "/Code/pycharmVersion/Data/Test/testBabelNetTotall_Mapped"
    pkl_file = open(outpath, 'wb')
    pickle.dump([uniqueSynIDBabelNetDict, test_docIndexLangTrans],pkl_file,-1)
    pkl_file.close()


    ##
def mapIndexLangTrans(docIndexLangTrans, mapFREN):
    for d in docIndexLangTrans:
        for indexLangTrans in docIndexLangTrans[d]:
            fr2EN = list()
            fr = indexLangTrans[1]['FR']
            if fr:
                for word in fr:
                    fr2EN.append("".join([mapFREN.get(w,w) for w in word]))
            indexLangTrans[1]['FR'] = fr2EN
    return docIndexLangTrans


    from pprint import pprint
    # eqCharas = [['À', 'à', 'Â', 'â', 'Æ', 'æ', 'Ä', 'ä'],
    #  ['Ç', 'ç'],
    #  ['É', 'é', 'È', 'è', 'Ê', 'ê', 'Ë', 'ë'],
    #  ['Î', 'î', 'Ï', 'ï'],
    #  ['Ô', 'ô', 'Œ', 'œ', 'Ö', 'ö'],
    #  ['Ù', 'ù', 'Û', 'û', 'Ü', 'ü'],
    #  ['Ÿ', 'ÿ']
    #  ]
    # enChara = ['a','c','e','i','o','u','y']
    # mapFREN = dict()
    # for i,e in enumerate(enChara):
    #     for f in eqCharas[i]:
    #         mapFREN[f]=e

    ##


##
if __name__ == "__main__":
    from DataMisc.loadPickleData import loadTrainData

    docFilteredIndexString, docIndexBabelSynsetID, docIndexLangTrans = loadTrainData()
    show_didnt_match_result(docFilteredIndexString, docIndexLangTrans)
    docIndexString_Lemma = lemmatizeFilteredIndexString(docFilteredIndexString,
                                                        docIndexLangTrans)
    show_didnt_match_result_after_lemma(docIndexString_Lemma, docIndexLangTrans)
