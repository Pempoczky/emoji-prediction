import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from typing import Union
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from fastapi import UploadFile, HTTPException


tt = TweetTokenizer()


def tokenize(text):
    return tt.tokenize(text)


st.title('Emoji Prediction')
if not os.path.isfile('models/vectorizer_model.joblib'):
    tfidfVectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words='english', token_pattern=None)
    train_text = open('data/train_text.txt', encoding="utf-8").read()
    lineSeparatedTrainText = train_text.splitlines()
    X = tfidfVectorizer.fit_transform(lineSeparatedTrainText)
    joblib.dump(tfidfVectorizer, 'models/vectorizer_model.joblib')
vectorizer_model = joblib.load('models/vectorizer_model.joblib')
svc_model = joblib.load('models/best_svc_model.joblib')
dt_classifier_model = joblib.load('models/best_dt_classifier_model.joblib')
knn_classifier_model = joblib.load('models/best_knn_classifier_model.joblib')
hard_ensemble_classifier = joblib.load('models/hard_ensemble_classifier.joblib')
soft_ensemble_classifier = joblib.load('models/soft_ensemble_classifier.joblib')


prompt = st.chat_input("Please input a short text, and I will reply with whichever emoji I think corresponds most to it!")
if prompt:
    try:
        # Preprocess the text
        vectorized_text = vectorizer_model.transform([prompt])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in text preprocessing: {str(e)}")
    # Make predictions using the loaded model
    prediction = svc_model.predict(vectorized_text)
    # prediction = dt_classifier_model.predict(vectorized_text)

    # Return the prediction
    # return {"prediction": prediction.tolist()}
    st.write(f"Prediction is: {prediction.toList()}")
# with st.chat_message("EmojiBot"):
#     st.write("Hi! Please input a short text, and I will reply with whichever emoji I think corresponds most to it!")
