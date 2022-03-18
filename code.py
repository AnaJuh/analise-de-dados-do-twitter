import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

vectorizer = CountVectorizer(analyzer = "word")

freq_tweets = vectorizer.fit_transform(tweets)

modelo = MultinomialNB()

modelo.fit(freq_tweets, classes)

testes = pd.read_csv("Tweets.csv")

freq_testes = vectorizer.transform(testes)

modelo.predict(freq_testes)
