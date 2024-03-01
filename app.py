

import streamlit as st
from transformers import pipeline

st.title('Article Sentiment Analysis')

# Load pre-trained sentiment analysis model
sentiment_analysis = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", revision="af0f99b")


# Create a text input for the user to enter the article
article = st.text_area('Enter your article here:')

# Perform sentiment analysis when the user clicks the button
if st.button('Analyze'):
    if not article:
        st.warning('Please enter an article.')
    else:
        sentiment = sentiment_analysis(article)[0]
        st.write('Sentiment:', sentiment['label'])
        st.write('Score:', sentiment['score'])
