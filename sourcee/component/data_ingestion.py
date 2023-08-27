import pandas as pd
import os
import sys
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sourcee.exception import CustomException
from sourcee.logger import logging
from sourcee.component.data_transformation import DataTransformation
from sourcee.component.model_trainer import ModelTraining
import warnings
warnings.filterwarnings("ignore")

@dataclass
class DataIngestionConfig:
    train_path = os.path.join('artifcats','train.csv')
    test_path = os.path.join('artifcats','test.csv')
    raw_path = os.path.join('artifcats','raw.csv')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def data_ingestion(self):
        logging.info('data ingestion started')
        try:
            df = pd.read_csv(os.path.join('notebook/data/adult.csv'))
            logging.info("dataset read")

            logging.info('train test split ')
            train_data,test_data = train_test_split(df,test_size=0.20,random_state=42)

            logging.info('make directory')
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_path),exist_ok=True)

            df.to_csv(self.data_ingestion_config.raw_path,index=False)
            train_data.to_csv(self.data_ingestion_config.train_path,index =False)
            test_data.to_csv(self.data_ingestion_config.test_path,index=False)

            logging.info('data ingestion completed')

            return(
                    self.data_ingestion_config.train_path,
                    self.data_ingestion_config.test_path

            )
        
            


        except Exception as e:
            logging.info('Error raised in data ingestion')
            raise CustomException(sys,e)

if __name__ == '__main__':
    obj = DataIngestion()
    train_data,test_data = obj.data_ingestion()
    trans = DataTransformation()
    train_arr,test_arr,_ = trans.initiate_data_transformation(train_data,test_data)
    train = ModelTraining()
    train.initiate_model_training(train_arr,test_arr)
    
    