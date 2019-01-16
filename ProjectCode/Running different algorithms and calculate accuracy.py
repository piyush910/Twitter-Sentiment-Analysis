
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

crossValidationData[0][0]
final_data = []
totalAccuracy = []
#for k in range(0,1):
for k in range(len(crossValidationData)):
    print("Cross Validation ------" + str(k))
    ktrainingData = crossValidationData[k][0][0]
    kAccuracyResults = []
    #Naive Bayesian Classification
    print("Naive Bayesian")
    classifier = nltk.NaiveBayesClassifier.train(ktrainingData)
    # Multinomial Naive Bayesian
    print("Multinomial Naive Bayesian")
    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(ktrainingData)
    # Bernoulli Naive Bayesian
    print("Bernoulli Naive Bayesian")
    BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
    BernoulliNB_classifier.train(ktrainingData)
    # Logistic Regression
    print("Logistic Regression")
    LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
    LogisticRegression_classifier.train(ktrainingData)
    # Linear SVC Classification
    print("Linear SVC Classification")
    LinearSVC_classifier = SklearnClassifier(LinearSVC())
    LinearSVC_classifier.train(ktrainingData)
    # NuSVC Classification
    print("NuSVC Classification")
    NuSVC_classifier = SklearnClassifier(NuSVC())
    NuSVC_classifier.train(ktrainingData)
    # Knearest Neighbors
    print("K nearest Neighbors Classification")
    Kn_classifier = SklearnClassifier(KNeighborsClassifier(20))
    Kn_classifier.train(ktrainingData)
    # Decision Tree Classification
    print("Decision Tree Classification")
    DecisionTree_classifier = SklearnClassifier(DecisionTreeClassifier(max_depth=5))
    DecisionTree_classifier.train(ktrainingData)
    # Random Forest Classification
    print("Random Forest Classification")
    RandomForest_classifier = SklearnClassifier(RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1))
    RandomForest_classifier.train(ktrainingData)
    # MLP(Neural Network Classification)
    #print("MLP - Neural Network Classification")
    #MLP_classifier = SklearnClassifier(MLPClassifier(alpha=1))
    #MLP_classifier.train(ktrainingData)
    # AdaBoost
    print("Adaboost Classification")
    adaBoost_classifier = SklearnClassifier(AdaBoostClassifier())
    adaBoost_classifier.train(ktrainingData)
   # Classifying test data
    print("Classifying test data")
    for record in crossValidationData[k][1]:
        final_data.append([record[0], record[1], classifier.classify(tweet_split(record[0])),
                           MNB_classifier.classify(tweet_split(record[0])),
                           BernoulliNB_classifier.classify(tweet_split(record[0])),
                           LogisticRegression_classifier.classify(tweet_split(record[0])),
                           LinearSVC_classifier.classify(tweet_split(record[0])),
                           NuSVC_classifier.classify(tweet_split(record[0])),
                           Kn_classifier.classify(tweet_split(record[0])),
                           DecisionTree_classifier.classify(tweet_split(record[0])),
                           RandomForest_classifier.classify(tweet_split(record[0])),
                           #MLP_classifier.classify(tweet_split(record[0])),
                           adaBoost_classifier.classify(tweet_split(record[0]))])
    for i in range(10):
        print("accuracy of classifier --- " + str(i))
        classifieraccuracy = []
        accfcst = 0
        totalTweets = 0
        truePos = 0
        trueNeg = 0
        actPos = 0
        actNeg = 0
        fcstPos = 0
        fcstNeg = 0
        for record in final_data:
            totalTweets+=1
            if record[1] == record[i+2]:
                accfcst+=1
                if record[i+2] == 1:
                    truePos+=1
                if record[i+2] == -1:
                    trueNeg+=1
            if record[1] == 1:
                actPos+=1
            if record[1] == -1:
                actNeg+=1
            if record[i+2] == 1:
                fcstPos+=1
            if record[i+2] == -1:
                fcstNeg+=1
        accuracy = accfcst / totalTweets
        if fcstPos == 0:
            posPrec = 0
        else:
            posPrec = truePos / fcstPos
        posRec = truePos / actPos
        if posPrec + posRec == 0:
            posFScore = 0
        else:
            posFScore = 2 * posPrec * posRec / (posPrec + posRec)
        if fcstNeg == 0:
            negPrec = 0
        else:
            negPrec = trueNeg / fcstNeg
        negRec = trueNeg / actNeg
        if negPrec + negRec == 0:
            negFScore = 0
        else:
            negFScore = 2 * negPrec * negRec / (negPrec + negRec)        
        classifierAccuracy = [accuracy, posPrec, posRec, negPrec, negRec, posFScore, negFScore]
        kAccuracyResults.append(classifierAccuracy)
    totalAccuracy.append(kAccuracyResults)