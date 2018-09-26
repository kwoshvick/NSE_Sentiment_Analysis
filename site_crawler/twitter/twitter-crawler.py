import tweepy

from site_crawler.twitter.credentials import Credentials


c = Credentials()

api = c.authentinticate_twitter()

public_tweets = api.home_timeline()
for tweet in public_tweets:
    print (tweet.text)
