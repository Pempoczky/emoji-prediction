import joblib
import os
from typing import Union

from nltk.tokenize import TweetTokenizer
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from starlette.responses import RedirectResponse
from fastapi import UploadFile, HTTPException

tt = TweetTokenizer()


def tokenize(text):
    return tt.tokenize(text)


app = FastAPI(
    title="Emoji Prediction Classifier",
    summary="An API endpoint to predict what emoji would fit best with a short piece of text. "
            "Trained on a dataset of US tweets and their corresponding emojis.",
    description="""
# An API endpoint to access an SVM
# Model usage
The model is trained on a dataset of US tweets, and so the input should roughly resemble a tweet for best results.
The input should not be more than 280 characters.

## Limitations
The model's dataset is imbalanced, so it may do better at predicting some emojis over others.

## Source
The model is sourced from https://github.com/Pempoczky/emoji-prediction.
    """,
    version="alpha",
)

# Load the trained classifier
if not os.path.isfile('models/vectorizer_model.joblib'):
    tfidfVectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words='english', token_pattern=None)
    train_text = open('data/train_text.txt', encoding="utf-8").read()
    lineSeparatedTrainText = train_text.splitlines()
    X = tfidfVectorizer.fit_transform(lineSeparatedTrainText)
    joblib.dump(tfidfVectorizer, 'models/vectorizer_model.joblib')
vectorizer_model = joblib.load('models/vectorizer_model.joblib')
svc_model = joblib.load('models/best_svc_model.joblib')


class TextInput(BaseModel):
    text: str

@app.get("/", description="Root endpoint that redirects to documentation.")
async def root():
    return RedirectResponse(url='/docs')

@app.post("/predict", description="Text classifier endpoint. Input text into the text field to send "
                                  "request. Text should not be more than 280 characters. "
                                  "Returns predicted class.")
async def predict(text_input: TextInput):
    text = text_input.text

    # vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words='english', token_pattern=None)
    vectorizer = vectorizer_model

    # Vectorize the preprocessed text using the same vectorizer as during training
    try:
        # Preprocess the text
        vectorized_text = vectorizer.transform([text])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in text preprocessing: {str(e)}")
    print(vectorized_text)
    # Make predictions using the loaded model
    prediction = svc_model.predict(vectorized_text)

    # Return the prediction
    return {"prediction": prediction.tolist()}
