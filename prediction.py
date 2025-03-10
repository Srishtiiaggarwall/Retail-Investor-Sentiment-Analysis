# 0 -> Negative, 1 -> Neutral, 2 -> Positive

import os
import sys
import pandas as pd
import re
import nltk
import emoji
import pickle
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from training import ModelTrainingConfig
from transformation import DictionaryConfig
from exception import CustomException
from logger import logging

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def process_text(text):
    try:
        text = text.strip()
        text = emoji.demojize(text)
        text = text.replace("_", " ")
        text = text.lower()
        text = text.replace("\n", " ")
        text = text.replace("  ", " ")
        stop_words = set(stopwords.words("english"))
        tokens = [word for word in text.split() if word not in stop_words]
        text = " ".join(tokens)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text
    except Exception as e:
        raise CustomException(e, sys)

def find_sentiment_score(text, positive_file, negative_file):
    with open(positive_file) as f1:
        pos_words = f1.readlines()
        pos_words = [word.lower().replace("\n", "") for word in pos_words]
        f1.close()

    with open(negative_file) as f2:
        neg_words = f2.readlines()
        neg_words = [word.lower().replace("\n", "") for word in neg_words]
        f2.close()

    tokens = wordpunct_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    pos_count = neg_count = 0
    for token in tokens:
        if token in pos_words:
            pos_count += 1
        elif token in neg_words:
            neg_count += 1
    sentiment_score = (pos_count - neg_count) / len(tokens)
    return sentiment_score


if __name__ == "__main__":
    # Loading the model
    model_path = ModelTrainingConfig.trained_model_file_path
    preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    model = load_object(file_path=model_path)
    preprocessor = load_object(file_path=preprocessor_path)

    # Input needed
    senti_label = "bearish"
    emo_label = "anger"

    positive_file = DictionaryConfig.positive_file
    negative_file = DictionaryConfig.negative_file

    # Examples
    positive_text = "Earnings report exceeded expectations! This stock is on fire! 🔥🔥 Time to celebrate! 🍾📊"
    neutral_text = "Some stocks up, some down… just another day in the market."
    negative_text = "Ditch everything today. No point in holding while the market keeps bleeding. Cashing out before I lose more. 😡💸 #BearMarket"

    text = process_text(negative_text)
    sentiment_score = find_sentiment_score(text, positive_file, negative_file)
    
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    pos_score = scores["pos"]
    neg_score = scores["neg"]
    neu_score = scores["neu"]
    compound = scores["compound"]


    mappings = {0:"Negative", 1:"Neutral", 2:"Positive"}
    data = {"emo_label":[emo_label], "senti_label":[senti_label], "processed":[text], "sentiment_score":[sentiment_score], "pos_score":[pos_score], "neg_score":[neg_score], "neu_score":[neu_score], "compound_score":[compound]}
    example = pd.DataFrame(data).reset_index(drop=True)
    transformed = preprocessor.transform(example)
    result = model.predict(transformed)
    print("Predicted : ", mappings[result[0]])

    # For the behaviourial insights
    import nltk
    from nltk.corpus import wordnet

    # nltk.download('wordnet')

    def get_synonyms(word, min_count=1000):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().replace('_', ' '))
        return list(synonyms)[:min_count]

    buying_words = ["buy", "purchase", "acquire", "invest", "bid", "accumulate"]
    selling_words = ["sell", "liquidate", "dump", "cash out", "exit", "unload"]
    holding_words = ["hold", "retain", "keep", "maintain", "wait", "sit tight"]

    buying_synonyms, selling_synonyms, holding_synonyms = set(), set(), set()

    for word in buying_words:
        buying_synonyms.update(get_synonyms(word, 1000))

    for word in selling_words:
        selling_synonyms.update(get_synonyms(word, 1000))

    for word in holding_words:
        holding_synonyms.update(get_synonyms(word, 1000))

    # print(len(buying_synonyms))
    # print(len(selling_synonyms))
    # print(len(holding_synonyms))

    def classify_behaviour(buy_score, sell_score):
        if buy_score - sell_score >= 1:
            return "Buying"
        elif buy_score - sell_score <= -1:
            return "Selling"
        else:
            return "Holding"

    sell_score = buy_score = 0
    print(text.split())
    for word in text.split():
        if word in buying_words or word in buying_synonyms:
            buy_score += 1
        elif word in selling_synonyms or word in selling_words:
            sell_score += 1

    print("Behaviour : ", classify_behaviour(buy_score, sell_score))