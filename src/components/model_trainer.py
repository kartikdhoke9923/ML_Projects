import os
import sys
from dataclasses import dataclass
from src.logger import logging
from src.utils import save_object, evaluate_model

import pandas as pd
import numpy as np
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report, roc_auc_score,accuracy_score)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier



class ModelTrainerConfig():
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr, preprocessor_path):
        try:
            logging.info("Splitting Training and testing data")
            X_train , y_train, X_test, y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]

            )
            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "XGBoost": XGBClassifier(),
                "LightGBM": LGBMClassifier()
            }
            params = {

                # ---------------- Logistic Regression ----------------
                "Logistic Regression": {
                    "C": np.logspace(-3, 2, 10),
                    "penalty": ["l2"],
                    "solver": ["lbfgs"],
                    "class_weight": [None, "balanced"],
                    "max_iter": [500, 1000]
                },

                # ---------------- Decision Tree ----------------
                "Decision Tree": {
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 10, 20],
                    "min_samples_leaf": [1, 5, 10],
                    "class_weight": [None, "balanced"]
                },

                # ---------------- Random Forest ----------------
                "Random Forest": {
                    "n_estimators": [200, 500, 800],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 10],
                    "min_samples_leaf": [1, 5],
                    "max_features": ["sqrt", "log2"],
                    "class_weight": [None, "balanced"]
                },

                # ---------------- Gradient Boosting ----------------
                "Gradient Boosting": {
                    "n_estimators": [200, 400],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5],
                    "subsample": [0.8, 1.0]
                },

                # ---------------- AdaBoost ----------------
                "AdaBoost": {
                    "n_estimators": [200, 400],
                    "learning_rate": [0.01, 0.05, 0.1]
                },

                # ---------------- XGBoost ----------------
                "XGBoost": {
                    "n_estimators": [300, 600],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0],
                    "gamma": [0, 1],
                    "scale_pos_weight": [1, 3, 5]  # 🔥 critical for imbalance
                },

                # ---------------- LightGBM ----------------
                "LightGBM": {
                    "n_estimators": [300, 600],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "num_leaves": [31, 63, 127],
                    "max_depth": [-1, 10, 20],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0],
                    "class_weight": [None, "balanced"]
                }
            }

            model_report = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models,
                                           params=params, n_iter=30, cv=3)
            
            for model_name, metrics in model_report.items():
                print(model_name)
                print("ROC-AUC :", metrics["roc_auc"])
                print("Recall  :", metrics["recall"])
                print("Precision:", metrics["precision"])
                print("F1-score:", metrics["f1_score"])
                print("-" * 40)
            best_model_name = max(
                model_report,
                key=lambda x: model_report[x]["roc_auc"]
            )

            best_model = model_report[best_model_name]["model"]

            print(f"Best model: {best_model_name}")
            print(f"Best ROC-AUC: {model_report[best_model_name]['roc_auc']}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            metrics_df = pd.DataFrame.from_dict(
                {
                    name: {
                        "roc_auc": v["roc_auc"],
                        "recall": v["recall"],
                        "precision": v["precision"],
                        "f1_score": v["f1_score"]
                    }
                    for name, v in model_report.items()
                },
                orient="index"
            )

            metrics_df.to_csv("artifacts/model_metrics.csv")
            return best_model_name, best_model
            


        except Exception as e:
            raise CustomException(e, sys)