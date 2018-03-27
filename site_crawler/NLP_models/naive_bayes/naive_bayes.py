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


def main():
    print("Hello Main method")
    X, y = readcsv()
    print("Experiment One")
    experiment1(X, y)  # call Different Experiments


if __name__ == "__main__":
    main()