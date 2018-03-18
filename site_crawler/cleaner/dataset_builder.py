import csv
from site_crawler.cleaner.cleaner import Cleaner

class Dataset_Builder:
    def __init__(self):
        self.cleaner = Cleaner()

    def write_tweet_txt(self, sentiment, name):
        file = open('../data/dataset/'+name, 'a')
        line = sentiment.strip()
        cleaned_line = self.cleaner.clean_tweets(line)
        file.write(cleaned_line)
        file.write('\n')

    def extract_sentiment_csv(self,csv_name):
        with open('../data/twitter_data/labeled_data/'+csv_name+'.csv', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # negative
                if row['label'] == '-1':
                    self.write_tweet_txt(row['text'].strip(), 'negative_sentiment.txt')
                # positive
                elif row['label'] == '1':
                    self.write_tweet_txt(row['text'].strip(), 'positive_sentiment.txt')
                # neutral / irrelevant
                elif row['label'] == '0':
                    self.write_tweet_txt(row['text'].strip(), 'neutral.txt')

if __name__ == "__main__":
    D_builder = Dataset_Builder()
    tweets_csvs = [
        '1'
    ]
    # tweets_csvs = [
    #     'Taifa_Leo',
    #     'BD_Africa',
    #     'RadioCitizenFM',
    #     'citizentvkenya',
    #     'KTNKenya',
    #     'K24Tv',
    #     'StandardKenya',
    #     'TheStarKenya',
    #     'radiomaisha',
    #     'KBCChannel1',
    #     'CapitalFMKenya',
    #     'African_Markets',
    #     'Africafinancial',
    #     'InvestInAfrica',
    #     'AfricanInvestor',
    #     'forbesafrica',
    #     'cnbcafrica',
    #     'BBCAfrica',
    #     'CNNAfrica',
    #     'allafrica',
    #     'ReutersAfrica',
    #     'VenturesAfrica',
    #     'BBGAfrica',
    #     'GhettoRadio895',
    #     'kenyanwalstreet',
    #     'SokoAnalyst',
    #     'NSEKenya',
    #     'wazua'
    # ]

    for tweets_csv in tweets_csvs:
        D_builder.extract_sentiment_csv(tweets_csv)










