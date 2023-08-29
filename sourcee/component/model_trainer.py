import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sourcee.logger import logging
from sourcee.exception import CustomException
from dataclasses import dataclass
from sourcee.utils import save_object, evaluate_models
import warnings
warnings.filterwarnings("ignore")

dataclass
class ModelTrainingConfig:
    trained_model_path = os.path.join('artifacts','model.pkl')

class ModelTraining:
    def __init__(self):
        self.model_training_config = ModelTrainingConfig()

    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info('Splitting Independent and Dependent features from train and test data')
            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
                )
            logging.info(X_train.dtype)
            print(X_test.dtype)
            print(y_train.dtype)
            print(y_test.dtype)

            
            logging.info('model training started')
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Logistic":LogisticRegression()
            }
            params = {
                "Random Forest":{
                    "class_weight":["balanced"],
                    'n_estimators': [20, 50, 30],
                    'max_depth': [10, 8, 5],
                    'min_samples_split': [2, 5, 10],
                },
                "Decision Tree":{
                    "class_weight":["balanced"],
                    "criterion":['gini',"entropy","log_loss"],
                    "splitter":['best','random'],
                    "max_depth":[3,4,5,6],
                    "min_samples_split":[2,3,4,5],
                    "min_samples_leaf":[1,2,3],
                    "max_features":["auto","sqrt","log2"]
                },
                "Logistic":{
                    "class_weight":["balanced"],
                    'penalty': ['l1', 'l2'],
                    #'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'solver': ['liblinear', 'saga']
                }
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                                models=models,param=params)

                ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f"Best Model Found, Model Name is: {best_model_name},Accuracy_Score: {best_model_score}")
            print("\n***************************************************************************************\n")
            logging.info(f"Best Model Found, Model Name is: {best_model_name},Accuracy_Score: {best_model_score}")

            save_object(
                file_path=self.model_training_config.trained_model_path,
                obj = best_model
                )

        except Exception as e:
            logging.info("Error Occured in Model Traning")
            raise CustomException(e,sys)