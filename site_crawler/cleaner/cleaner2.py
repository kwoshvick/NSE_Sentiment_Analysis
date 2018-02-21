import re

m = 'RT @FrontPagesKe Learn the fundamentals of a "Cash Flow Statement" https://t.co/H0SNvWzKk4 #IMLeducate https://t.co/f0LFl0BwAH @kwoshvick #manahe_tyh'

m = m.lower()
# text = re.sub(r'^https?:\/\/.*[\r\n]*', '', m, flags=re.MULTILINE)

n = re.sub(r'http\S+', '', m) #url
o = re.sub(r'#\w*','',n) # hastag
p = re.sub(r'@\w*\s?','',o) #username
q = p.replace("rt","",True) # remove to retweet

print(q)
