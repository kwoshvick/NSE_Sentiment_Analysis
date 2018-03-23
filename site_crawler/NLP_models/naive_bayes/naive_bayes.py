import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn import datasets
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn import svm

# Declare the categories
categories = ['Crime', 'Family']

# Load the dataset
docs_to_train = sklearn.datasets.load_files("/Users/danielhoadley/Documents/Development/Python/Test_Data", description=None, categories=categories,
                                            load_content=True, shuffle=True, encoding='utf-8', decode_error='strict', random_state=0)

train_X, test_X, train_y, test_y = train_test_split(docs_to_train.data,
                               docs_to_train.target,
                               test_size = 3)
print (len(docs_to_train.data))

print (train_X)

# Vectorise the dataset

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(docs_to_train.data)

# Fit the estimator and transform the vector to tf-idf

tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
X_train_tf.shape


tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

# Train the naive Bayes classifier

clf = MultinomialNB().fit(X_train_tfidf, docs_to_train.target)

docs_new = ['The defendant used a knife.', 'This court will protect vulnerable adults', 'The appellant was sentenced to seven years']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

# Print the results

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, docs_to_train.target_names[category]))