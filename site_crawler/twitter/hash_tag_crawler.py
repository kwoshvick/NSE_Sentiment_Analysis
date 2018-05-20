import tweepy
from site_crawler.twitter.credentials import Credentials
from site_crawler.cleaner.cleaner import Cleaner
import csv
import pandas as pd
from sklearn.externals import joblib
model=joblib.load('model.pkl')

credentials = Credentials()
cleaner = Cleaner()
api = credentials.authentinticate_twitter()

def predict(text2):

    from sklearn.externals import joblib

    model = joblib.load('model.pkl')

    prediction = model.predict(text2)

    return prediction[0]



text2 = [
    "the world's smallest disneyland has posted losses for 9 of the 12 years since it opened. local visitors make up 41%… ",
    "kenya's economy struggles",
    "loss making venture",
    "Uchumi",
    "nakumatt",
    "Centum ",
    "use becomes a public limited company"
    ]



query = 'safaricom'
max_tweets = 10


searched_tweets = [status for status in tweepy.Cursor(api.search, q=query).items(max_tweets)]


outtweets = [[cleaner.clean_tweets(tweet.text),predict([cleaner.clean_tweets(tweet.text)])] for tweet in searched_tweets]


# print(outtweets)
#
# exit()

# for tweets in outtweets:
#     print([tweets])

with open('./predict.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["text", "label"])
    writer.writerows(outtweets)
pass

# df=pd.read_csv("./predict.csv")
# df=df.dropna(how='any')
# df=df.drop_duplicates()
# model=joblib.load("model.pkl")
# print(df.text)
# df['label'] = model.predict(df.text)
# print(df.label)
# df.to_csv("predicted.csv",encoding="utf8")
# #print(model.predict(df.text))
# print("read")