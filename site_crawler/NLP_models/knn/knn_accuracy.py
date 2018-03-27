# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 17:30:01 2017

@author: Kaari
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_predict
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn import metrics

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import string


def readcsv():
    df = pd.read_csv("../data/dataset/csv/dataset_sentiment.csv", )  # read labelled tweets
    # df2=df.reindex(np.random.permutation(df.index))
    X = df.text
    y = df.label
    return X, y


def createSVM(X, y):
    svm_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('svm', SVC(kernel="linear", C=1))])
    svm_clf = svm_clf.fit(X, y)
    return svm_clf


def createNB(X, y):
    nb_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('nb', MultinomialNB())])
    nb_clf = nb_clf.fit(X, y)
    return nb_clf


def drawrocSVM(y_test, y_pred):
    fpr, tpr, threshold = roc_curve(y_test, y_pred)
    print("Drawing")
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='SVM AUC = %0.2f' % roc_auc, color='b')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def drawrocNB(y_test, y_pred):
    fpr, tpr, threshold = roc_curve(y_test, y_pred)
    print("Drawing")
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='NB AUC = %0.2f' % roc_auc, color='r')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def drawrocKNN(y_test, y_pred):
    fpr, tpr, threshold = roc_curve(y_test, y_pred)
    print("Drawing")
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='KNN AUC = %0.2f' % roc_auc, color='g')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def experiment1(X, y):
    """Different Classifiers"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    # SVM classifier
    svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('svm', SVC(kernel="linear", C=1))])
    svm = svm.fit(X_train, y_train)
    ypred = svm.predict(X_test)
    print("SVM metrics")
    print(metrics.accuracy_score(y_test, ypred))
    print(metrics.classification_report(y_test, ypred))
    drawrocSVM(y_test, ypred)
    # NB classifier
    nb = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('nb', MultinomialNB())])
    nb = nb.fit(X_train, y_train)
    yprednb = nb.predict(X_test)
    print("NB Metrics")
    print(metrics.accuracy_score(y_test, yprednb))
    print(metrics.classification_report(y_test, yprednb))
    drawrocNB(y_test, yprednb)
    # KNN classifier
    knn = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('knn', KNeighborsClassifier())])
    knn = knn.fit(X_train, y_train)
    ypredknn = knn.predict(X_test)
    print("KNN evaluation")
    print(metrics.accuracy_score(y_test, ypredknn))
    print(metrics.classification_report(y_test, ypredknn))
    drawrocKNN(y_test, ypredknn)


def experiment2(X, y):
    """Different features with SVM"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    svm = createSVM(X_train, y_train)
    y_pred = (svm.predict(X_test))
    print("Original Accuracy: Unigram with tf-idf")
    print(metrics.confusion_matrix(y_test, y_pred))
    print(metrics.accuracy_score(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred))
    svm2 = Pipeline([('vect', CountVectorizer()), ('svm', SVC(kernel="linear", C=1))])
    svm2 = svm2.fit(X_train, y_train)
    ypred2 = svm2.predict(X_test)
    print("Just unigram counts Accuracy")
    print(metrics.accuracy_score(y_test, ypred2))
    print(metrics.classification_report(y_test, ypred2))
    svm3 = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))), ('svm', SVC(kernel="linear", C=1))])
    svm3 = svm3.fit(X_train, y_train)
    ypred3 = svm3.predict(X_test)
    print("just bigram counts Accuracy")
    print(metrics.accuracy_score(y_test, ypred3))
    print(metrics.classification_report(y_test, ypred3))
    svm4 = Pipeline([('vect', CountVectorizer(ngram_range=(1, 3))), ('svm', SVC(kernel="linear", C=1))])
    svm4 = svm4.fit(X_train, y_train)
    ypred4 = svm4.predict(X_test)
    print("Trigram counts Accuracy")
    print(metrics.accuracy_score(y_test, ypred4))
    print(metrics.classification_report(y_test, ypred4))
    svm5 = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))), ('tfidf', TfidfTransformer()),
                     ('svm', SVC(kernel="linear", C=1))])
    svm5 = svm5.fit(X_train, y_train)
    ypred5 = svm5.predict(X_test)
    print("bigram with tfidf Accuracy")
    # print(metrics.confusion_matrix(y_test,ypred5))
    print(metrics.accuracy_score(y_test, ypred5))
    print(metrics.classification_report(y_test, ypred5))
    svm6 = Pipeline([('vect', CountVectorizer(ngram_range=(1, 3))), ('tfidf', TfidfTransformer()),
                     ('svm', SVC(kernel="linear", C=1))])
    svm6 = svm6.fit(X_train, y_train)
    ypred6 = svm6.predict(X_test)
    print("trigram with tfidf Accuracy")
    # print(metrics.confusion_matrix(y_test,ypred6))
    print(metrics.accuracy_score(y_test, ypred6))
    print(metrics.classification_report(y_test, ypred6))


def experiment3(X, y):
    """Different Feature set with Naive Bayes"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    nb = createNB(X_train, y_train)
    y_pred = nb.predict(X_test)
    print("Original Accuracy")
    print(metrics.classification_report(y_test, y_pred))
    print(metrics.accuracy_score(y_test, y_pred))
    nb2 = Pipeline([('vect', CountVectorizer()), ('nb', MultinomialNB())])
    nb2 = nb2.fit(X_train, y_train)
    ypred2 = nb2.predict(X_test)
    print("Just counts Accuracy")
    print(metrics.classification_report(y_test, ypred2))
    print(metrics.accuracy_score(y_test, ypred2))
    nb3 = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))), ('nb', MultinomialNB())])
    nb3 = nb3.fit(X_train, y_train)
    ypred3 = nb3.predict(X_test)
    print("bigram counts Accuracy")
    print(metrics.accuracy_score(y_test, ypred3))
    print(metrics.classification_report(y_test, ypred3))
    nb4 = Pipeline([('vect', CountVectorizer(ngram_range=(1, 3))), ('nb', MultinomialNB())])
    nb4 = nb4.fit(X_train, y_train)
    ypred4 = nb4.predict(X_test)
    print("Trigram counts Accuracy")
    print(metrics.accuracy_score(y_test, ypred4))
    print(metrics.classification_report(y_test, ypred4))
    nb5 = Pipeline(
        [('vect', CountVectorizer(ngram_range=(1, 2))), ('tfidf', TfidfTransformer()), ('nb', MultinomialNB())])
    nb5 = nb5.fit(X_train, y_train)
    ypred5 = nb5.predict(X_test)
    # drawrocSVM(y_test,ypred5)
    print("bigram with tfidf Accuracy")
    print(metrics.accuracy_score(y_test, ypred5))
    print(metrics.classification_report(y_test, ypred5))
    nb6 = Pipeline(
        [('vect', CountVectorizer(ngram_range=(1, 3))), ('tfidf', TfidfTransformer()), ('nb', MultinomialNB())])
    nb6 = nb6.fit(X_train, y_train)
    ypred6 = nb6.predict(X_test)
    # drawrocSVM(y_test,ypred5)
    print("trigram with tfidf Accuracy")
    print(metrics.classification_report(y_test, ypred6))
    print(metrics.accuracy_score(y_test, ypred6))


def experiment4(X, y):
    """Different feature sets with KNN"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    knn = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('knn', KNeighborsClassifier())])
    knn = knn.fit(X_train, y_train)
    ypredknn = knn.predict(X_test)
    print("Original Accuracy: Unigram tfidf")
    print(metrics.accuracy_score(y_test, ypredknn))
    print(metrics.classification_report(y_test, ypredknn))

    knn = Pipeline([('vect', CountVectorizer()), ('knn', KNeighborsClassifier())])
    knn = knn.fit(X_train, y_train)
    ypredknn = knn.predict(X_test)
    print("Unigram counts")
    print(metrics.accuracy_score(y_test, ypredknn))
    print(metrics.classification_report(y_test, ypredknn))

    knn = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))), ('knn', KNeighborsClassifier())])
    knn = knn.fit(X_train, y_train)
    ypredknn = knn.predict(X_test)
    print("Bigram counts")
    print(metrics.accuracy_score(y_test, ypredknn))
    print(metrics.classification_report(y_test, ypredknn))

    knn = Pipeline(
        [('vect', CountVectorizer(ngram_range=(1, 2))), ('tfidf', TfidfTransformer()), ('knn', KNeighborsClassifier())])
    knn = knn.fit(X_train, y_train)
    ypredknn = knn.predict(X_test)
    print("Bigram tfidf")
    print(metrics.accuracy_score(y_test, ypredknn))
    print(metrics.classification_report(y_test, ypredknn))

    knn = Pipeline([('vect', CountVectorizer(ngram_range=(1, 3))), ('knn', KNeighborsClassifier())])
    knn = knn.fit(X_train, y_train)
    ypredknn = knn.predict(X_test)
    print("trigram counts")
    print(metrics.accuracy_score(y_test, ypredknn))
    print(metrics.classification_report(y_test, ypredknn))

    knn = Pipeline(
        [('vect', CountVectorizer(ngram_range=(1, 3))), ('tfidf', TfidfTransformer()), ('knn', KNeighborsClassifier())])
    knn = knn.fit(X_train, y_train)
    ypredknn = knn.predict(X_test)
    print("Trigram tfidf")
    print(metrics.accuracy_score(y_test, ypredknn))
    print(metrics.classification_report(y_test, ypredknn))


def main():
    print("Hello Main method")
    X, y = readcsv()
    print("Experiment One")
    experiment1(X, y)  # call Different Experiments


if __name__ == "__main__":
    main()