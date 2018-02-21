import re
import string

m = 'RT ?  ! hhh&amprr! @FrontPagesKe Learn the fundamentals of a "Cash Flow Statement" https://t.co/H0SNvWzKk4 #IMLeducate https://t.co/f0LFl0BwAH @kwoshvick #manahe_tyh'

m = m.lower()
# text = re.sub(r'^https?:\/\/.*[\r\n]*', '', m, flags=re.MULTILINE)

translator = str.maketrans('', '', string.punctuation)

n = re.sub(r'http\S+', '', m) #url
o = re.sub(r'#\w*','',n) # hastag
p = re.sub(r'@\w*\s?','',o) #username
q = p.replace("rt","",True) # remove to retweet
r = q.replace(['&amp','&gt'],[""],True)

print(r)

# s = r.replace("&gt","",True)
#
# t = s.translate(translator)
#
#
#
# print(t)

# print
# import csv
# with open('test.csv', newline='', encoding='utf-8') as csvfile:
#     reader = csv.DictReader(csvfile)
#     for row in reader:
#         print(row[''])
#         # text = row['text']
#         # print(text)
#         # cleaned_text = self.clean_tweets(text)