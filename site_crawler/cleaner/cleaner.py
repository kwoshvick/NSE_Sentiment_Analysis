import csv
import re

class Cleaner:

    def read_csv(self,csv_name):
        with open(csv_name, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                text = row['text']
                cleaned_text = self.clean_tweets(text)

    def clean_tweets(self,tweet):
        # harmonize the cases
        lower_case_text = tweet.lower()
        # remove urls
        removed_url = re.sub(r'http\S+', '', lower_case_text)
        # remove hashtags
        removed_hash_tag = re.sub(r'#\w*', '', removed_url)  # hastag
        # remove usernames from tweets
        removed_username = re.sub(r'@\w*\s?','',removed_hash_tag)
        # removed retweets
        removed_retweet = removed_username.replace("rt", "", True)  # remove to retweet
        # removing punctuations
        # remove non utf8 characters

        return









