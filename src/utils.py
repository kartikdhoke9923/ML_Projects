import os
import sys
import dill
import numpy as np 
import pandas as pd
import pickle

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (confusion_matrix, classification_report,recall_score,f1_score, roc_auc_score,precision_score,accuracy_score)
from sklearn.base import ClassifierMixin

from src.exception import CustomException



def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(
    X_train,
    y_train,
    X_test,
    y_test,
    models: dict,
    params: dict,
    n_iter=25,
    cv=3
):
    """
    Train multiple classification models using RandomizedSearchCV
    and evaluate using ROC-AUC + Recall.
    """
    try:
        report = {}

        for model_name, model in models.items():
            print(f"\nTraining {model_name} with RandomizedSearchCV...")

            param_dist = params.get(model_name, {})

            rs = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_dist,
                n_iter=n_iter,
                scoring="roc_auc",        # primary optimisation metric
                cv=cv,
                n_jobs=-1,
                random_state=42,
                verbose=1
            )

            rs.fit(X_train, y_train)

            best_model = rs.best_estimator_

            # Predictions
            y_pred = best_model.predict(X_test)
            y_proba = best_model.predict_proba(X_test)[:, 1]

            # Metrics
            roc_auc = roc_auc_score(y_test, y_proba)
            recall = recall_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            report[model_name] = {
                "model": best_model,
                "roc_auc": roc_auc,
                "recall": recall,
                "precision": precision,
                "f1_score": f1,
                "best_params": rs.best_params_
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)