import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("diamonds.csv")
df.columns = df.columns.str.capitalize()

# Define the feature list explicitly
FEATURE_COLUMNS = ['Carat', 'Cut', 'Color', 'Clarity', 'Depth', 'Table']

# Ensure only the defined features are used
X = df[FEATURE_COLUMNS]
y = df['Price']

# Handle missing values
X = X.fillna(0)
y = y.fillna(0)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a pre-tuned CatBoost model (default parameters)
pre_tuned_model = CatBoostRegressor(cat_features=['Cut', 'Color', 'Clarity'], verbose=0)
pre_tuned_model.fit(X_train, y_train)
pre_tuned_model.save_model("pre_tuned_catboost_model.cbm")

# Train a tuned CatBoost model
tuned_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=3,
    bagging_temperature=1,
    cat_features=['Cut', 'Color', 'Clarity'],
    verbose=0
)
tuned_model.fit(X_train, y_train)
tuned_model.save_model("tuned_catboost_model.cbm")

print("Pre-tuned and tuned CatBoost models have been trained and saved successfully!")

tuned_model.fit(X_train, y_train)
tuned_model.save_model("tuned_catboost_model.cbm")

print("Pre-tuned and tuned CatBoost models have been trained and saved successfully!")
