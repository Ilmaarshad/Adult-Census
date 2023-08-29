import os
import sys
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sourcee.exception import CustomException
from sourcee.logger import logging
from dataclasses import dataclass
from sourcee.utils import save_object
import warnings
warnings.filterwarnings("ignore")

dataclass
class DataTransformationConfig:
    pickle_path = os.path.join('artifcats','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def data_transformation(self):
        try:
            logging.info('data transformation started')

            logging.info('seperate categorical and numerical feature')

            categorical_feature = ['workclass', 'education', 'marital-status', 'occupation',
                                        'relationship', 'race', 'sex', 'country']
            
            numerical_feature = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
                                        'hours-per-week']
            
            logging.info('pipeline initiated')

            num_pipeline = Pipeline([
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline([
                ("imputer",SimpleImputer(strategy='most_frequent')),
                ('encode',OrdinalEncoder()),
                ('onehot',OneHotEncoder(sparse=False,handle_unknown="ignore")),
                ('scaler',StandardScaler())
            ])

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline,numerical_feature),
                ('cat_pipeline',cat_pipeline,categorical_feature)
            ])

            return preprocessor
        
        
        except Exception as e:
            logging.info("error raised in data transformation")
            raise CustomException(e,sys)
        

    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Reading train and test data completed')

            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocesser_obj = self.data_transformation()

            train_feature_df = train_df.drop(columns='salary',axis=1)
            target_train_df = train_df['salary']

            test_feature_df = test_df.drop(columns='salary',axis=1)
            target_test_df = test_df['salary']

            logging.info("Applying preprocessing object on training and testing datasets.")

            train_feature_arr = preprocesser_obj.fit_transform(train_feature_df)
            test_feature_arr = preprocesser_obj.transform(test_feature_df)


            train_arr = np.c_[train_feature_arr, np.array(target_train_df)]
            test_arr = np.c_[test_feature_arr, np.array(target_test_df)]

            logging.info(f'Data type : {type(train_arr)}')


            save_object(
                file_path = self.data_transformation_config.pickle_path,
                obj=preprocesser_obj

            )
            logging.info('pickle file saved')
            

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.pickle_path
            )

        except Exception as e:
            logging.info('Error raise in initiate data transformation')
            raise CustomException(e,sys)