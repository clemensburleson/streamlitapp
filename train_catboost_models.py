#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:08:45 2024

@author: clemensburleson
"""

import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("diamonds.csv")
df.columns = df.columns.str.capitalize()

# Prepare features and target
X = df.drop(columns=['Price'], errors='ignore')
y = df['Price']

# Handle missing values
X = X.fillna(0)
y = y.fillna(0)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a pre-tuned CatBoost model (default parameters)
pre_tuned_model = CatBoostRegressor(cat_features=list(X.select_dtypes(include=['object']).columns), verbose=0)
pre_tuned_model.fit(X_train, y_train)
pre_tuned_model.save_model("pre_tuned_catboost_model.cbm")

# Define hyperparameters for tuning
tuned_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=3,
    bagging_temperature=1,
    cat_features=list(X.select_dtypes(include=['object']).columns),
    verbose=0
)
tuned_model.fit(X_train, y_train)
tuned_model.save_model("tuned_catboost_model.cbm")

print("Pre-tuned and tuned CatBoost models have been trained and saved successfully!")
