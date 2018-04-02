import os

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

for i in tweets_csvs:
    checker = os.path.exists('unlabeled_'+i+'.csv')
    if not checker:
        print(i)



