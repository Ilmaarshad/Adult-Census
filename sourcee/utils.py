import os
import sys
import pickle
import numpy as np 
import pandas as pd
from sourcee.exception import CustomException
from sourcee.logger import logging
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


#save pickle file
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        logging.info("Error raised in save_object in utils")
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            #Model Traning
            gs = GridSearchCV(estimator=model,param_grid=para,cv=5)
            gs.fit(X_train,y_train)
            model(**gs.best_params_).fit(X_train,y_train)
            #model.fit(X_train,y_train)

            #make Prediction
            y_test_pred = model.predict(X_test)

            test_model_score = accuracy_score(y_test,y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        logging.info("error raised in evaluate models")
        raise CustomException(e, sys)