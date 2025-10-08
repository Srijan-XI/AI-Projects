# src/model_training.py
from sklearn.linear_model import LinearRegression
import joblib

def train_model(X_train, y_train, model_path="linear_regression_model.pkl"):
    model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    return model
