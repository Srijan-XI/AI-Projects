# titanic_survival_prediction/main.py
from src.data_preprocessing import load_data, clean_data, encode_features
from src.model_training import train_and_evaluate
import os


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    df = load_data("data/titanic.csv")
    if df.empty:
        raise ValueError("The dataset is empty. Please check the file path and content.")    

    # Clean data
    df = clean_data(df)

    # Detect correct target column
    target_col = 'survived' if 'survived' in df.columns else 'Survived'

    # Split features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Encode categorical features
    X, encoder = encode_features(X)

    # Train and evaluate model
    train_and_evaluate(X, y)
