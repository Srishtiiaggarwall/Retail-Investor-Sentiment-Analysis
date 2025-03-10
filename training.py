import sys
import os
import numpy as np
import pickle
from dataclasses import dataclass
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from tqdm.auto import tqdm
from ingestion import DataIngestion
from transformation import DataTransformer
from logger import logging
from exception import CustomException

@dataclass
class ModelTrainingConfig:
    trained_model_file_path : str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainingConfig()

    def initiate_model_training(self, X_train, Y_train, X_test, Y_test):
        try:
            logging.info("Initiating Model Trainer")
            
            models = {
                "DecisionTree":DecisionTreeClassifier(),
                "RandomForest":RandomForestClassifier(),
                "AdaBoost":AdaBoostClassifier(),
                "GradientBoost":GradientBoostingClassifier(),
                "XGBoost":XGBClassifier()
            }

            params = {
                "DecisionTree":{
                    "criterion": ["gini", "log_loss"],
                    "max_depth": [2, 8, 16, 24]
                },

                "RandomForest":{
                    "n_estimators":[100, 250, 500],
                    "criterion": ["gini", "log_loss"],
                    "max_depth": [2, 8, 16, 24]
                },

                "AdaBoost":{
                    "n_estimators":[100, 250, 500],
                    "learning_rate":[1, .1, .01, 0.5,.001]
                },

                "GradientBoost":{
                    "n_estimators":[100, 250, 500],
                    "learning_rate": [1, .1, .01, 0.5,.001]
                },

                "XGBoost":{
                    "learning_rate":[.1,.01,.05,.001],
                    "n_estimators": [100, 250, 500]
                }
            }

            model_report : dict = self.evaluate_models(
                X_train=X_train,
                Y_train=Y_train,
                X_test=X_test,
                Y_test=Y_test,
                models=models,
                param=params
            )

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")

            print("Model Training Completed")
            logging.info(f"Best found model on both training and testing dataset")
            
            self.save_object(
                file_path=self.config.trained_model_file_path,
                obj=best_model
            )
            print("Best Model Saved")

        except Exception as e:
            raise CustomException(e, sys)

    def save_object(self, file_path, obj):
        try:
            dir_path = os.path.dirname(file_path)

            os.makedirs(dir_path, exist_ok=True)

            with open(file_path, "wb") as file_obj:
                pickle.dump(obj, file_obj)

        except Exception as e:
            raise CustomException(e, sys)
    
    def evaluate_models(self, X_train, Y_train, X_test, Y_test, models, param):
        try:
            report = {}

            for i in tqdm(range(len(list(models)))):
                model = list(models.values())[i]
                para=param[list(models.keys())[i]]
                gs = GridSearchCV(model,para,cv=5)
                gs.fit(X_train,Y_train)
                model.set_params(**gs.best_params_)
                model.fit(X_train,Y_train)
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                train_model_score = f1_score(Y_train, y_train_pred, average="weighted")
                test_model_score = f1_score(Y_test, y_test_pred, average="weighted")
                report[list(models.keys())[i]] = test_model_score
            
            return report

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    
    data = DataIngestion(test=True)
    train_data = data.train_data
    val_data = data.val_data
    test_data = data.test_data

    train_transformer = DataTransformer(data=train_data, name="Training")
    val_transformer = DataTransformer(data=val_data, name="Validation")
    test_transformer = DataTransformer(data=test_data, name="Testing")

    X_train, Y_train = train_transformer.transformed_data
    X_val, Y_val = val_transformer.transformed_data
    X_test, Y_test = test_transformer.transformed_data

    trainer = ModelTrainer()
    trainer.initiate_model_training(X_train, Y_train, X_test, Y_test)
    