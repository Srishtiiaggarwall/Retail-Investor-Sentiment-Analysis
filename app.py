import os
import sys
import pandas as pd
import re
import nltk
import emoji
import pickle
import streamlit as st
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")
nltk.download('vader_lexicon')
from training import ModelTrainingConfig
from transformation import DictionaryConfig
from exception import CustomException
from logger import logging
from prediction import load_object, process_text, find_sentiment_score


model_path = ModelTrainingConfig.trained_model_file_path
preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

model = load_object(file_path=model_path)
preprocessor = load_object(file_path=preprocessor_path)

emotions = ["optimism", "anxiety", "excitement","disgust", "belief", "ambiguous", "amusement", "confusion", "anger", "panic", "surprise", "depression"] 
market_label = ["bearish", "bullish"]

positive_file = DictionaryConfig.positive_file
negative_file = DictionaryConfig.negative_file

st.set_page_config(page_title="Retail-Investor-Sentiment-Analysis", layout="centered")
st.title("📈 Retail Investor Sentiment Analysis")

st.sidebar.header("User Input")
selected_emotion = st.sidebar.selectbox("Select Emotion", emotions)
selected_market = st.sidebar.selectbox("Select Market Type", market_label)

tweet_text = st.text_area("Enter Tweet Text:", placeholder="Type the investor's tweet here...")

def split_hashtag(hashtag):
    words = re.findall(r'[A-Za-z][a-z]*', hashtag)
    return " ".join(words)

def predict_sentiment(tweet_text):
    text = split_hashtag(tweet_text)
    text = process_text(text)
    tweet_text = process_text(tweet_text)
    sentiment_score = find_sentiment_score(tweet_text, positive_file, negative_file)
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(tweet_text)
    pos_score = scores["pos"]
    neg_score = scores["neg"]
    neu_score = scores["neu"]
    compound = scores["compound"]
    mappings = {0:"Negative", 1:"Neutral", 2:"Positive"}
    data = {"emo_label":[selected_emotion], "senti_label":[selected_market], "processed":[tweet_text], "sentiment_score":[sentiment_score], "pos_score":[pos_score], "neg_score":[neg_score], "neu_score":[neu_score], "compound_score":[compound]}
    example = pd.DataFrame(data).reset_index(drop=True)
    transformed = preprocessor.transform(example)
    result = model.predict(transformed)
    sentiment = mappings[result[0]]
    return sentiment, text

def get_synonyms(word, min_count=1000):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().replace('_', ' '))
        return list(synonyms)[:min_count]

buying_words = ["buy", "purchase", "acquire", "invest", "bid", "accumulate", "bull"]
selling_words = ["sell", "liquidate", "dump", "cash out", "exit", "unload", "bear"]
holding_words = ["hold", "retain", "keep", "maintain", "wait", "sit tight", "no move"]

buying_synonyms, selling_synonyms, holding_synonyms = set(), set(), set()

for word in buying_words:
    buying_synonyms.update(get_synonyms(word, 1000))

for word in selling_words:
    selling_synonyms.update(get_synonyms(word, 1000))

for word in holding_words:
    holding_synonyms.update(get_synonyms(word, 1000))

def classify_behaviour(buy_score, sell_score, hold_score):
    if hold_score >= buy_score and hold_score >= sell_score:
        return "Holding"
    elif buy_score - sell_score >= 1:
        return "Buying"
    elif buy_score - sell_score <= -1:
        return "Selling"
    else:
        return "Holding"


if st.button("Analyze Sentiment"):
    if tweet_text.strip():
        sentiment, text = predict_sentiment(tweet_text)
        sell_score = buy_score = hold_score = 0
        for word in text.split():
            if word in buying_words or word in buying_synonyms:
                buy_score += 1
            elif word in holding_synonyms or word in holding_words:
                hold_score += 1
            elif word in selling_synonyms or word in selling_words:
                print(word)
                sell_score += 1
            
        # print(buy_score, sell_score, hold_score)
        behaviour = classify_behaviour(buy_score, sell_score, hold_score)
        st.success(f"Predicted Sentiment: **{sentiment}**")
        st.success(f"Investor Behaviourial Insight : **{behaviour}**")
    else:
        st.warning("Please enter some text to analyze sentiment.")