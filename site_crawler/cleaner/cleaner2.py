import re
import string
import html
from html.parser import HTMLParser

m = ' ;      &gtRT ? ;     ;      ! hhh&amprr! @FrontPagesKe Learn the fundamentals of a "Cash Flow Statement" https://t.co/H0SNvWzKk4 #IMLeducate https://t.co/f0LFl0BwAH @kwoshvick #manahe_tyh'

x = m.split()

z = ' '.join(x)

n = z.replace(';','')

# re.sub(',','', s)
print(n)

# m = "b'RT @MwananchiNews: Israel yataka wahamiaji Waafrika waondoke&gt	&gt	&gt	https://t.co/h1RDix0jIH https://t.co/j5JOCF28WX'"
# p = html.unescape(m)
#
# print(p[2:][:-1])

# print(p.decode('utf8').encode('ascii','ignore'))

# m = m.lower()
# # text = re.sub(r'^https?:\/\/.*[\r\n]*', '', m, flags=re.MULTILINE)
#
# translator = str.maketrans('', '', string.punctuation)
#
# n = re.sub(r'http\S+', '', m) #url
# o = re.sub(r'#\w*','',n) # hastag
# p = re.sub(r'@\w*\s?','',o) #username
# q = p.replace("rt","",True) # remove to retweet
# r = q.replace(['&amp','&gt'],[""],True)
#
# print(r)

# s = r.replace("&gt","",True)
#
# t = s.translate(translator)
#
#
#
# print(t)

# print
# import csv
# with open('Taifa_Leo2.csv', encoding='utf-8',) as csvfile:
#     reader = csv.DictReader(csvfile)
#     for row in reader:
#         a = row['text']
#         print(a)
#         a = html.unescape(a)
#         #
#         print(a)
        # b = a[2:][:-1]
        # print(b.replace("[^\x00-\x7F]+","",True))
        # text = row['text']
        # print(text)
        # cleaned_text = self.clean_tweets(text)