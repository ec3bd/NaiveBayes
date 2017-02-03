#Eamon Collins    ec3bd
#!/usr/bin/python

import sys
import os
import re
import numpy as np
import math
from sklearn.naive_bayes import MultinomialNB

###############################################################################

#Unkown in index 0
vocab = dict.fromkeys(['love', 'loves', 'loving','loved'], 1)
vocab.update(dict.fromkeys(['wonderful', 'wonderfully'], 2))
vocab.update(dict.fromkeys(['best', 'bests'], 3))
vocab.update(dict.fromkeys(['great', 'greater', 'greats'], 4))
vocab.update(dict.fromkeys(['superb', 'superbly'], 5))
vocab.update(dict.fromkeys(['still'], 6))
vocab.update(dict.fromkeys(['beautiful', 'beautifully', 'beauteous'], 7))
vocab.update(dict.fromkeys(['bad', 'badly'], 8))
vocab.update(dict.fromkeys(['worst', 'worse', 'worsts'], 9))
vocab.update(dict.fromkeys(['stupid', 'stupidest', 'stupidly'], 10))
vocab.update(dict.fromkeys(['waste', 'wastes', 'waster', 'wasted', 'wasting'], 11))
vocab.update(dict.fromkeys(['boring', 'bore', 'bored', 'bores'], 12))
vocab.update(dict.fromkeys(['?'], 13))
vocab.update(dict.fromkeys(['!'], 14))

#So I realized afterwards that this was not how it was supposed to be stemmed,
#so uncomment the section below to use the standard vocabulary

vocab = dict.fromkeys(['love', 'loves', 'loving','loved'], 1)
vocab['wonderful'] = 2
vocab['best'] = 3
vocab['great'] = 4
vocab['superb'] = 5
vocab['still'] = 6
vocab['beautiful'] = 7
vocab['bad'] = 8
vocab['worst'] = 9
vocab['stupid'] = 10
vocab['waste'] = 11
vocab['boring'] = 12
vocab.update(dict.fromkeys(['?'], 13))
vocab.update(dict.fromkeys(['!'], 14))


def transfer(fileDj, vocabulary):
    fileVect = [0 for x in range(15)]
    fileTokens = re.findall("[\w']+|[?!]", fileDj)
    for token in fileTokens:
        if token in vocab:
            fileVect[vocabulary[token.lower()]] += 1
        else:
            fileVect[0] += 1
    return fileVect


def loadData(Path):
    xTrain = []
    xTest = []
    yTrain = []
    yTest = []
    for filename in os.listdir(Path+"training_set/pos/"):
        file = open(Path+"training_set/pos/"+filename, 'r')
        xTrain.append(transfer(file.read(), vocab))
        file.close()
        yTrain.append(1)
    for filename in os.listdir(Path+"training_set/neg/"):
        file = open(Path+"training_set/neg/"+filename, 'r')
        xTrain.append(transfer(file.read(), vocab))
        file.close()
        yTrain.append(-1)
    for filename in os.listdir(Path+"test_set/pos/"):
        file = open(Path+"test_set/pos/"+filename, 'r')
        xTest.append(transfer(file.read(), vocab))
        file.close()
        yTest.append(1)
    for filename in os.listdir(Path+"test_set/neg/"):
        file = open(Path+"test_set/neg/"+filename, 'r')
        xTest.append(transfer(file.read(), vocab))
        file.close()
        yTest.append(-1)

    return np.asmatrix(xTrain), np.asmatrix(xTest), np.asmatrix(yTrain).T, np.asmatrix(yTest).T


def naiveBayesMulFeature_train(xTrain, yTrain):
    thetaPos = []
    thetaNeg = []
    numFeats = xTrain.shape[1]
    for i in range(0, numFeats):
        thetaPos.append(0)
        thetaNeg.append(0)
    for i in range(0, len(xTrain)):
        for j in range(0, numFeats):
            if yTrain[i] == 1:
                thetaPos[j] += xTrain[i,j]
            if yTrain[i] == -1:
                thetaNeg[j] += xTrain[i,j]

    posFeatures = sum(thetaPos)
    negFeatures = sum(thetaNeg)
    #The +1 and +posFeatures are Laplace smoothing so nothing has a 0 probability
    for j in range(0, numFeats):
        thetaPos[j] = (thetaPos[j] + 1) / (posFeatures + len(xTrain[0])/2)#(thetaPos[j] + thetaNeg[j])
        thetaNeg[j] = (thetaNeg[j] + 1) / (negFeatures + len(xTrain[0])/2)#(tempThetaPos[j] + thetaNeg[j])
    return np.asmatrix(thetaPos).T, np.asmatrix(thetaNeg).T


def naiveBayesMulFeature_test(Xtest, ytest,thetaPos, thetaNeg):
    yPredict = []
    numFeats = thetaPos.shape[0]
    numSamps = Xtest.shape[0]
    posChance = [0 for i in range(numSamps)]
    negChance = [0 for i in range(numSamps)]
    for j in range(numSamps):
        for i in range(numFeats):
            posChance[j] += math.log(pow(thetaPos[i,0], Xtest[j,i]))
            negChance[j] += math.log(pow(thetaNeg[i,0], Xtest[j,i]))

    for i in range(numSamps):
        if posChance[i] > negChance[i]:
            yPredict.append(1)
        else:
            yPredict.append(-1)

    true = 0
    total = 0
    for i in range(numSamps):
        if yPredict[i] == ytest[i,0]:
            true += 1
        total += 1
    Accuracy = true/total

    return yPredict, Accuracy


def naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest):
    clf = MultinomialNB()
    clf.fit(Xtrain, ytrain)
    Accuracy = clf.score(Xtest, ytest)
    return Accuracy



def naiveBayesMulFeature_testDirectOne(path, thetaPos, thetaNeg, vocabulary):
    p = thetaPos.shape[0]
    file = open(path, 'r')
    fileText = file.read()
    fileVect = [0 for x in range(15)]
    fileTokens = re.findall("[\w']+|[?!]", fileText)
    for token in fileTokens:
        if token in vocab:
            fileVect[vocabulary[token.lower()]] += 1
        else:
            fileVect[0] += 1
    posChance = 0
    negChance = 0
    for i in range(p):
        posChance += math.log(pow(thetaPos[i,0], fileVect[i]))
        negChance += math.log(pow(thetaNeg[i,0], fileVect[i]))
    if posChance > negChance:
        return 1
    else:
        return -1


def naiveBayesMulFeature_testDirect(path,thetaPos, thetaNeg, vocabulary):
    yPredict = []
    total = 0
    posCorr = 0
    negCorr = 0
    for filename in os.listdir(path+"pos/"):
        predict = naiveBayesMulFeature_testDirectOne(path+"pos/"+filename, thetaPos, thetaNeg, vocabulary)
        yPredict.append(predict)
        total += 1
        if(predict == 1): posCorr += 1
    for filename in os.listdir(path+"neg/"):
        predict = naiveBayesMulFeature_testDirectOne(path+"neg/"+filename, thetaPos, thetaNeg, vocabulary)
        yPredict.append(predict)
        total += 1
        if(predict == -1): negCorr += 1
    Accuracy = (negCorr + posCorr) / total

    return yPredict, Accuracy



def naiveBayesBernFeature_train(Xtrain, ytrain):
    n = Xtrain.shape[0]
    p = Xtrain.shape[1]
    thetaPosTrue = [0 for i in range(p)]
    thetaNegTrue = [0 for i in range(p)]
    for i in range(n):
        for j in range(p):
            if ytrain[i] == 1 and Xtrain[i,j] > 0:
                thetaPosTrue[j] += 1
            elif ytrain[i] == -1 and Xtrain[i,j] > 0:
                thetaNegTrue[j] += 1

    for j in range(p):
        thetaPosTrue[j] = (thetaPosTrue[j] + 1) / (n / 2 + 2)
        thetaNegTrue[j] = (thetaNegTrue[j] + 1) / (n / 2 + 2)
    return thetaPosTrue, thetaNegTrue


def naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue):
    yPredict = []
    n = Xtest.shape[0]
    p = Xtest.shape[1]
    posChance = [0 for i in range(n)]
    negChance = [0 for i in range(n)]
    for i in range(n):
        for j in range(p):
            posChancej = thetaPosTrue[j]
            negChancej = thetaNegTrue[j]
            if Xtest[i,j] == 0:
                posChancej = 1 - thetaPosTrue[j]
                negChancej = 1 - thetaNegTrue[j]
            posChance[i] += math.log(posChancej)
            negChance[i] += math.log(negChancej)

    for i in range(n):
        if posChance[i] > negChance[i]:
            yPredict.append(1)
        else:
            yPredict.append(-1)

    true = 0
    total = 0
    for i in range(n):
        if yPredict[i] == ytest[i,0]:
            true += 1
        total += 1
    Accuracy = true/total
    return yPredict, Accuracy


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print ("Usage: python naiveBayes.py dataSetPath testSetPath")
        sys.exit()

    print ("--------------------")
    textDataSetsDirectoryFullPath = sys.argv[1]
    testFileDirectoryFullPath = sys.argv[2]


    Xtrain, Xtest, ytrain, ytest = loadData(textDataSetsDirectoryFullPath)


    thetaPos, thetaNeg = naiveBayesMulFeature_train(Xtrain, ytrain)
    print ("thetaPos =", thetaPos)
    print ("thetaNeg =", thetaNeg)
    print ("--------------------")

    yPredict, Accuracy = naiveBayesMulFeature_test(Xtest, ytest, thetaPos, thetaNeg)
    print ("MNBC classification accuracy =", Accuracy)

    Accuracy_sk = naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest)
    print ("Sklearn MultinomialNB accuracy =", Accuracy_sk)

    yPredict, Accuracy = naiveBayesMulFeature_testDirect(testFileDirectoryFullPath, thetaPos, thetaNeg, vocab)
    print ("Directly MNBC tesing accuracy =", Accuracy)
    print ("--------------------")

    thetaPosTrue, thetaNegTrue = naiveBayesBernFeature_train(Xtrain, ytrain)
    print ("thetaPosTrue =", thetaPosTrue)
    print ("thetaNegTrue =", thetaNegTrue)
    print ("--------------------")

    yPredict, Accuracy = naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue)
    print ("BNBC classification accuracy =", Accuracy)
    print ("--------------------")
