import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

st.title("Sentiment Prediction of Review")

review = st.text_input("Enter your review")

word_index = imdb.get_word_index()

reverse_word_index = {}

for k,v in word_index.items():
    reverse_word_index[v] = k

global model
model = load_model("./model_files/simple_rnn_imdb.h5")

def preprocess_text(review):
    words = review.lower().split()
    encoded_review = [word_index.get(word, 2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen = 500)
    return padded_review


def predict_sentiment(review):
    padded_review = preprocess_text(review)
    pred = model.predict(padded_review)

    sentiment = "Positive" if pred[0][0] >= 0.5 else "Negative"

    return sentiment, pred[0][0]

sentiment, score = predict_sentiment(review)

st.write(f"Sentiment of review: {sentiment}, with confidence score: {score}")