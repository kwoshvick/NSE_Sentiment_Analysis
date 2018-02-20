import csv
import tweepy
from site_crawler.twitter.credentials import Credentials


class Account_All_Tweets:

    def __init__(self):
        credentials = Credentials()
        self.api = credentials.authentinticate_twitter()


    def get_all_tweets(self,profile_name):
            # Twitter only allows access to a users most recent 3240 tweets with this method

            # initialize a list to hold all the tweepy Tweets
            alltweets = []

            # make initial request for most recent tweets (200 is the maximum allowed count)
            new_tweets = self.api.user_timeline(screen_name=profile_name, count=200)

            # save most recent tweets
            alltweets.extend(new_tweets)

            # save the id of the oldest tweet less one
            oldest = alltweets[-1].id - 1

            # keep grabbing tweets until there are no tweets left to grab
            while len(new_tweets) > 0:
                print("getting tweets before %s") % (oldest)
                # all subsiquent requests use the max_id param to prevent duplicates
                new_tweets = self.api.user_timeline(screen_name=profile_name, count=200, max_id=oldest)
                # save most recent tweets
                alltweets.extend(new_tweets)
                # update the id of the oldest tweet less one
                oldest = alltweets[-1].id - 1
                print("...%s tweets downloaded so far" % (len(alltweets)))

            # transform the tweepy tweets into a 2D array that will populate the csv
            outtweets = [[tweet.id_str, tweet.created_at, tweet.text.encode("utf-8")] for tweet in alltweets]
            # write the csv
            with open('%s_tweets.csv' % profile_name, 'wb') as f:
                writer = csv.writer(f)
                writer.writerow(["id", "created_at", "text"])
                writer.writerows(outtweets)
            pass
