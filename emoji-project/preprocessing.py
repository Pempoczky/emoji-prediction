import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.tokenize import LineTokenizer, TweetTokenizer

# mapping = open('data/mapping.txt', encoding="utf-8").read()
# train_labels = open('data/train_labels.txt', encoding="utf-8").read()
train_text = open('data/train_text.txt', encoding="utf-8").read()
lt = LineTokenizer()
tt = TweetTokenizer()

def tokenize(text):
    return tt.tokenize(text)

vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words='english', token_pattern=None)
lineSeparatedText = lt.tokenize(train_text)
X = vectorizer.fit_transform(lineSeparatedText)
# print(vectorizer.get_feature_names_out()[30500:30600])

#save the array of features as a binary file in the features folder
np.save('features/feature_data.npy', X)
