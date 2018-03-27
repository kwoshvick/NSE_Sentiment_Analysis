import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def readcsv():
    df = pd.read_csv("../../data/dataset/csv/dataset_sentiment.csv", )  # read labelled tweets
    X = df.text
    y = df.label
    return X, y


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


def knn_accuracy(X, y):
    """Different Classifiers"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    knn = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('knn', KNeighborsClassifier())])
    knn = knn.fit(X_train, y_train)
    ypredknn = knn.predict(X_test)
    print("KNN evaluation")
    print(metrics.accuracy_score(y_test, ypredknn))
    print(metrics.classification_report(y_test, ypredknn))
    drawrocKNN(y_test, ypredknn)

def main():
    X, y = readcsv()
    knn_accuracy(X, y)


if __name__ == "__main__":
    main()