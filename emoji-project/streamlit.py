import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from typing import Union
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from fastapi import UploadFile, HTTPException
from preprocessing import tokenize


def label_to_emoji(label):
    if label == 0:
        return "❤️"
    elif label == 1:
        return "😍"
    elif label == 2:
        return "😂"
    elif label == 3:
        return "💕"
    elif label == 4:
        return "🔥"
    elif label == 5:
        return "😊"
    elif label == 6:
        return "😎"
    elif label == 7:
        return "✨"
    elif label == 8:
        return "💙"
    elif label == 9:
        return "😘"
    elif label == 10:
        return "📷"
    elif label == 11:
        return "🇺🇸"
    elif label == 12:
        return "☀"
    elif label == 13:
        return "💜"
    elif label == 14:
        return "😉"
    elif label == 15:
        return "💯"
    elif label == 16:
        return "😁"
    elif label == 17:
        return "🎄"
    elif label == 18:
        return "📸"
    elif label == 19:
        return "😜"
    else:
        return "Invalid label"


st.title('Emoji Prediction')
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
    input = prompt
    # Make predictions using the loaded model
    svc_prediction = svc_model.predict(vectorized_text)
    dt_prediction = dt_classifier_model.predict(vectorized_text)
    knn_prediction = knn_classifier_model.predict(vectorized_text)
    hard_ensemble_prediction = hard_ensemble_classifier.predict(vectorized_text)
    soft_ensemble_prediction = soft_ensemble_classifier.predict(vectorized_text)

    st.write(f"Text input: {input}")
    # Return the predictions
    # return {"prediction": prediction.tolist()}
    st.write(f"Support Vector Classifier's prediction is: {label_to_emoji(svc_prediction[0])}")
    st.write(f"Decision Tree Classifier's prediction is: {label_to_emoji(dt_prediction[0])}")
    st.write(f"K Nearest Neighbours Classifier's prediction is: {label_to_emoji(knn_prediction[0])}")
    st.write(f"Hard Ensemble Classifier's prediction is: {label_to_emoji(hard_ensemble_prediction[0])}")
    st.write(f"Soft Ensemble Classifier's prediction is: {label_to_emoji(soft_ensemble_prediction[0])}")



# with st.chat_message("EmojiBot"):
#     st.write("Hi! Please input a short text, and I will reply with whichever emoji I think corresponds most to it!")
