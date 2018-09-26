from sklearn.externals import joblib
model=joblib.load('model.pkl')

text2 = [
    # "the world's smallest disneyland has posted losses for 9 of the 12 years since it opened. local visitors make up 41%… ",
    #      "kenya's economy struggles",
         "loss making venture"
    #      "Uchumi",
    #      "nakumatt",
    #      "Centum ",
    #      "use becomes a public limited company"
         ]

a =model.predict(text2)

print(a[0])