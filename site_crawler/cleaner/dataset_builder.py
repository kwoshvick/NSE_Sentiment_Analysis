import csv
from site_crawler.cleaner.cleaner import Cleaner

class Dataset_Builder:
    def __init__(self):
        self.cleaner = Cleaner()
        self.create_csv_headers()

    def create_csv_headers(self):
        csv_files = [
            'negative_sentiment',
            'positive_sentiment',
            'dataset_sentiment'
        ]
        for csv_file in csv_files:
            with open('../data/dataset/csv/' + csv_file + '.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(["text", "label"])

    def write_tweet_txt(self, sentiment, name):
        file = open('../data/dataset/txt/'+name+ '.txt', 'a')
        line = sentiment.strip()
        cleaned_line = self.cleaner.clean_tweets(line)
        file.write(cleaned_line)
        file.write('\n')

    def write_tweet_csv(self, sentiment, name, polarity):
        with open('../data/dataset/csv/' + name + '.csv', 'a') as f:
            writer = csv.writer(f)
            line = sentiment.strip()
            cleaned_line = self.cleaner.clean_tweets(line)
            writer.writerow([cleaned_line,polarity, ])
        pass

    def extract_sentiment_csv(self,csv_name):
        with open('../data/twitter_data/labeled_data/unlabeled_'+csv_name+'.csv', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # negative
                if row['label'] == '-1':
                    self.write_tweet_txt(row['text'].strip(), 'negative_sentiment.txt')
                    self.write_tweet_csv(row['text'].strip(), 'negative_sentiment','-1')
                    self.write_tweet_csv(row['text'].strip(), 'dataset_sentiment','-1')
                # positive
                elif row['label'] == '1':
                    self.write_tweet_txt(row['text'].strip(), 'positive_sentiment')
                    self.write_tweet_csv(row['text'].strip(), 'positive_sentiment', '1')
                    self.write_tweet_csv(row['text'].strip(), 'dataset_sentiment', '1')
                # neutral / irrelevant
                elif row['label'] == '0':
                    self.write_tweet_txt(row['text'].strip(), 'neutral')

if __name__ == "__main__":
    D_builder = Dataset_Builder()

    tweets_csvs = [
        'Business_KE',
        'MadeItInAfrica',
        'IFCAfrica',
        'africareview',
        'AfDB_Group',
        '_AfricanUnion',
        'Taifa_Leo',
        'BD_Africa',
        'RadioCitizenFM',
        'citizentvkenya',
        'KTNKenya',
        'K24Tv',
        'StandardKenya',
        'TheStarKenya',
        'radiomaisha',
        'KBCChannel1',
        'CapitalFMKenya',
        'African_Markets',
        'Africafinancial',
        'InvestInAfrica',
        'AfricanInvestor',
        'forbesafrica',
        'cnbcafrica',
        'BBCAfrica',
        'CNNAfrica',
        'allafrica',
        'ReutersAfrica',
        'VenturesAfrica',
        'BBGAfrica',
        'GhettoRadio895',
        'kenyanwalstreet',
        'SokoAnalyst',
        'NSEKenya',
        'wazua'
    ]

    for tweets_csv in tweets_csvs:
        D_builder.extract_sentiment_csv(tweets_csv)










