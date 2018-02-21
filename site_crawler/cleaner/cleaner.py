import csv
import re
import string

class Cleaner:

    def __init__(self):
        self.remove_punctuations = str.maketrans('', '', string.punctuation)

    def read_csv(self,csv_name):
        cleaned_text = []
        with open(csv_name, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                text = row['text']
                # print(text)
                a = self.clean_tweets(text)
                # print(a)
                cleaned_text.append(a)

        print(cleaned_text)

        self.save_cleaned_csv('me',cleaned_text)

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
        # removed_punctuation = removed_retweet.translate(self.remove_punctuations)
        # print(removed_username)
        # remove spaces
        remove_g_t = removed_username.replace("&gt", "", True)
        final_text = remove_g_t.replace("&amp", "", True)

        n = final_text.split()

        return final_text


    def save_cleaned_csv(self,name,tweets_list):
        print(tweets_list)
        with open('../data/twitter_data/cleaned_data/' + name + '.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["text"])
            for tweet in tweets_list:
                # wr.writerow([item, ])
                writer.writerow([tweet,])
        pass


if __name__ == "__main__":
    c = Cleaner()
    c.read_csv('Africafinancial.csv')










