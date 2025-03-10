import os
import sys
import pandas as pd

from dataclasses import dataclass
from logger import logging
from exception import CustomException

@dataclass
class DataConfig:
    train_file : str = os.path.join("Data", "twitter_data_train.csv")
    val_file : str = os.path.join("Data", "twitter_data_val.csv")
    test_file : str = os.path.join("Data", "twitter_data_test.csv")

class DataIngestion:
    def __init__(self, train=True, val=True, test=False):
        try:

            self.__config = DataConfig()
            logging.info("Data Ingestion Initiated.")
            self.train_data = self.initiate_ingestion(self.__config.train_file, train)
            self.val_data = self.initiate_ingestion(self.__config.val_file, val)
            self.test_data = self.initiate_ingestion(self.__config.test_file, test)
            logging.info("Data Ingestion Completed.")
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_ingestion(self, file_path: str, flag: bool) -> pd.DataFrame:
        if not flag:
            return pd.DataFrame(data=None)
        try:
            data = pd.read_csv(file_path)
            return data
        except Exception as e:
            raise CustomException(e, sys)