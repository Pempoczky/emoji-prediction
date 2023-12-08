import numpy as np
#from nltk.tokenize import LineTokenizer, TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import LineTokenizer, TweetTokenizer
from tempfile import TemporaryFile

outfile = TemporaryFile()


mapping = open('data/mapping.txt', encoding="utf-8").read()
train_labels = open('data/train_labels.txt', encoding="utf-8").read()
train_text = open('data/train_text.txt', encoding="utf-8").read()
lt = LineTokenizer()
tt = TweetTokenizer()

def tokenize(text):
    return tt.tokenize(text)

vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words='english', token_pattern=None)
tokentext = lt.tokenize(train_text)

X = vectorizer.fit_transform(tokentext)
#save the array of features as a binary file in the features folder
np.save('features/feature_data.npy', X)
