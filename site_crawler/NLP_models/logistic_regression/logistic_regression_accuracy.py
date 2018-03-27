import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer




def readcsv():
    df = pd.read_csv("../../data/dataset/csv/dataset_sentiment.csv", )  # read labelled tweets
    X = df.text
    y = df.label
    return X, y

def logistic_regression_accuracy(X, y):
    """Different Classifiers"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    logreg = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('logistic', LogisticRegression(C=.001, multi_class='multinomial',solver='lbfgs',random_state=1))])
    logreg = logreg.fit(X_train, y_train)
    ypred = logreg.predict(X_test)
    print("Logistic Regression metrics")
    print(metrics.accuracy_score(y_test, ypred))
    print(metrics.classification_report(y_test, ypred))





def main():
    X, y = readcsv()
    logistic_regression_accuracy(X, y)


if __name__ == "__main__":
    main()