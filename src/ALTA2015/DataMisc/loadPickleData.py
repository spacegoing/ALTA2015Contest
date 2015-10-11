import pickle


def loadTrainData():
    inputpath = "/Users/spacegoing/百度云同步盘/macANU/" \
                "2cdSemester 2015/Document Analysis/sharedTask" \
                "/Code/pycharmVersion/Data/Train/trainBabelfy"
    pkl_file = open(inputpath, 'rb')
    docIndexString, docStringIndices, docIndexBabelSynsetID, \
    docFilteredIndexString = pickle.load(pkl_file)
    pkl_file.close()

    inputpath = "/Users/spacegoing/百度云同步盘/macANU/" \
                "2cdSemester 2015/Document Analysis/sharedTask" \
                "/Code/pycharmVersion/Data/Train/trainBabelNet"
    pkl_file = open(inputpath, 'rb')
    uniqueSynIDBabelNetDict, docIndexLangTrans = pickle.load(pkl_file)
    pkl_file.close()

    return docFilteredIndexString, docIndexBabelSynsetID, docIndexLangTrans


def loadTestData():
    inputpath = "/Users/spacegoing/百度云同步盘/macANU/" \
                "2cdSemester 2015/Document Analysis/sharedTask" \
                "/Code/pycharmVersion/Data/Test/testBabelfy"
    pkl_file = open(inputpath, 'rb')
    docIndexString, docStringIndices, docIndexBabelSynsetID, \
    docFilteredIndexString = pickle.load(pkl_file)
    pkl_file.close()

    inputpath = "/Users/spacegoing/百度云同步盘/macANU/" \
                "2cdSemester 2015/Document Analysis/sharedTask" \
                "/Code/pycharmVersion/Data/Test/testBabelNetTotall"
    pkl_file = open(inputpath, 'rb')
    uniqueSynIDBabelNetDict, docIndexLangTrans = pickle.load(pkl_file)
    pkl_file.close()

    return docFilteredIndexString, docIndexBabelSynsetID, docIndexLangTrans
    # test_docFilteredIndexString, test_docIndexBabelSynsetID, test_docIndexLangTrans

