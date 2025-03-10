import os
import sys
import re
import nltk
import emoji
import pandas as pd
import numpy as np
import pickle
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
# nltk.download("punkt")
# nltk.download("stopwords")
from dataclasses import dataclass
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from exception import CustomException
from logger import logging

@dataclass
class DictionaryConfig:
    try:
        positive_file : str = os.path.join("Dictionary", "Positive-words.txt")
        negative_file : str = os.path.join("Dictionary", "Negative-words.txt")
    except Exception as e:
        raise CustomException(e, sys) 

class DataTransformer:
    def __init__(self, data=None, name=""):
        try:
            self.__data = data
            self.__config = DictionaryConfig()
            logging.info(f"{name} Data Transformation Initiated.")
            self.transformed_data = self.transform(
                data=self.__data,
                positive_file=self.__config.positive_file,
                negative_file=self.__config.negative_file
            )
            logging.info(f"{name} Data Transformation Completed.")
        except Exception as e:
            raise CustomException(e, sys)

    def clean(self, data : pd.DataFrame) -> pd.DataFrame:
        try:
            data.drop(columns=["id", "date", "ticker"], axis=1, inplace=True)
            data.drop_duplicates(inplace=True)
            data.dropna(axis=1, how="all", inplace=True)
            return data
        except Exception as e:
            raise CustomException(e, sys)
    
    def preprocess_text(self, text : str) -> str:
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

    def classify_sentiment(self, score : float) -> str:
        try:
            if score > 0.2:
                return "Positive"
            elif score < 0:
                return "Negative"
            else:
                return "Neutral"  
        except Exception as e:
            raise CustomException(e, sys)

    def feature_engineering(self, data : pd.DataFrame) -> np.array:
        try:
            text_features = ["original", "processed", "label"]
            input_feature = "processed"
            output_feature = "label"
            categorical_cols = [col for col in data.columns if col not in text_features and data.dtypes[col] == "object"]
            numerical_cols = [col for col in data.columns if data.dtypes[col] != "object"]

            ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="error")
            scaler = StandardScaler()
            le = LabelEncoder()
            vectorizer = TfidfVectorizer()
            pca = PCA(n_components=10)

            encoded_output = le.fit_transform(data[output_feature])

            preprocessor = ColumnTransformer([
                ("vectorize", vectorizer, input_feature),
                ("ohe", ohe, categorical_cols),
                ("scaler", scaler, numerical_cols),
            ], remainder="drop")

            pipeline = Pipeline([
                ("preprocessing", preprocessor),
                ("dimensionality_reduction", pca)
            ])

            pipeline.fit(data)
            with open("artifacts/preprocessor.pkl", "wb") as f:
                pickle.dump(pipeline, f)
            logging.info("Saved Pipeline at artifacts/preprocessor.pkl")
            
            processed_data = pipeline.transform(data)
            
            return processed_data, encoded_output.reshape(-1, 1)
        
        except Exception as e:
            raise CustomException(e, sys)

    def transform(self, data : pd.DataFrame, positive_file : str, negative_file : str) -> np.array:
        try:
            if data is None:
                return np.array(data)
            
            ### Data Cleaning ### 
            print("Before Cleaning , Data shape : ", data.shape)
            data = self.clean(data)
            print("After Cleaning , Data shape : ", data.shape)
            
            ### Data Transformation ###
            print("Preprocessing Data : ")
            data["processed"] = ""
            data["sentiment_score"] = 0.0
            data["neg_score"] = 0.0
            data["pos_score"] = 0.0
            data["neu_score"] = 0.0
            data["compound_score"] = 0.0
            data["label"] = ""
            
            sia = SentimentIntensityAnalyzer()
            
            with open(positive_file) as f1:
                pos_words = f1.readlines()
                pos_words = [word.lower().replace("\n", "") for word in pos_words]
                f1.close()

            with open(negative_file) as f2:
                neg_words = f2.readlines()
                neg_words = [word.lower().replace("\n", "") for word in neg_words]
                f2.close()

            for i, row in tqdm(data.iterrows(), total=data.shape[0]):
                tweet = row["original"]
                processed_tweet = self.preprocess_text(tweet)
                scores = sia.polarity_scores(processed_tweet)
                data.loc[i, "processed"] = processed_tweet
                data.loc[i, "compound_score"] = scores["compound"]
                data.loc[i, "pos_score"] = scores["pos"]
                data.loc[i, "neg_score"] = scores["neg"]
                data.loc[i, "neu_score"] = scores["neu"]
                
                tokens = wordpunct_tokenize(processed_tweet)
                stop_words = set(stopwords.words("english"))
                tokens = [word for word in tokens if word not in stop_words]

                pos_count = neg_count = 0
                for token in tokens:
                    if token in pos_words:
                        pos_count += 1
                    elif token in neg_words:
                        neg_count += 1
                sentiment_score = (pos_count - neg_count) / len(tokens)
                data.loc[i, "sentiment_score"] = sentiment_score 
                data.loc[i, "label"] = self.classify_sentiment(sentiment_score)

            print("After Preprocessing , Data shape : ", data.shape)
            ### Feature Engineering ###
            X, Y = self.feature_engineering(data)
            print(f"After Feature Engineering, X shape : {X.shape} and Y shape : {Y.shape}")
            return X, Y.reshape(-1)
            
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from ingestion import DataIngestion
    data = DataIngestion()
    train_data = data.train_data

    transformer = DataTransformer(data=train_data, name="Training")
    X_train, Y_train = transformer.transformed_data