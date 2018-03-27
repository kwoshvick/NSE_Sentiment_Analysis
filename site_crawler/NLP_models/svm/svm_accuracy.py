import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def readcsv():
    df = pd.read_csv("../../data/dataset/csv/dataset_sentiment.csv", )  # read labelled tweets
    X = df.text
    y = df.label
    return X, y

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


def svm_accuracy(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
    svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('svm', SVC(kernel="linear", C=1))])
    svm = svm.fit(X_train, y_train)
    ypred = svm.predict(X_test)
    print("SVM metrics")
    print(metrics.accuracy_score(y_test, ypred))
    print(metrics.classification_report(y_test, ypred))
    drawrocSVM(y_test, ypred)

def main():
    X, y = readcsv()
    svm_accuracy(X, y)


if __name__ == "__main__":
    main()