import pandas as pd
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def readcsv():
    df = pd.read_csv("../../data/dataset/csv/dataset_sentiment.csv", )
    X = df.text
    y = df.label
    return X, y

def createNB(X, y):
    nb_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('nb', MultinomialNB())])
    nb_clf = nb_clf.fit(X, y)
    return nb_clf

def naive_bayes_ngram(X, y):
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


def main():
    X, y = readcsv()
    naive_bayes_ngram(X, y)


if __name__ == "__main__":
    main()