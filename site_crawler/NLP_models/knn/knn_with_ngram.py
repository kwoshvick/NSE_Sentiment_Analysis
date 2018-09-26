import pandas as pd
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def readcsv():
    df = pd.read_csv("../../data/dataset/csv/dataset_sentiment.csv", )
    X = df.text
    y = df.label
    return X, y


def knn_ngram(X, y):
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
    X, y = readcsv()
    knn_ngram(X, y)

if __name__ == "__main__":
    main()