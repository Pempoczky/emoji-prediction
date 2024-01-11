import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import TweetTokenizer
from scipy.sparse import save_npz
import joblib


train_text = open('data/train_text.txt', encoding="utf-8").read()
val_text = open('data/val_text.txt', encoding="utf-8").read()  # Corrected the path for validation data
tt = TweetTokenizer()

# Separate lines before tokenization
lineSeparatedTrainText = train_text.splitlines()
lineSeparatedValText = val_text.splitlines()


def tokenize(text):
    return tt.tokenize(text)


vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words='english', token_pattern=None)
# Tokenize and fit_transform on training data
X_train = vectorizer.fit_transform(lineSeparatedTrainText)

# Tokenize and transform on validation data
X_val = vectorizer.transform(lineSeparatedValText)

# Save the sparse matrices
save_npz('features/train_feature_data.npz', X_train)
save_npz('features/val_feature_data.npz', X_val)
# joblib.dump(vectorizer, 'models/vectorizer_model.joblib')
