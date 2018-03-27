import pandas as pd
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def readcsv():
    df = pd.read_csv("../../data/dataset/csv/dataset_sentiment.csv", )
    X = df.text
    y = df.label
    return X, y


def createSVM(X, y):
    svm_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('svm', SVC(kernel="linear", C=1))])
    svm_clf = svm_clf.fit(X, y)
    return svm_clf




def svm_ngram(X, y):
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



def main():
    X, y = readcsv()
    svm_ngram(X, y)  # call Different Experiments


if __name__ == "__main__":
    main()