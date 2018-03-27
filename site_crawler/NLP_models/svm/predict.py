from sklearn.externals import joblib
model=joblib.load('model.pkl')

text2 = ["the world's smallest disneyland has posted losses for 9 of the 12 years since it opened. local visitors make up 41%… ",
         "kenya's economy struggles",
         "loss making venture",
         "Uchumi",
         "nakumatt"
         ]

print(model.predict(text2))