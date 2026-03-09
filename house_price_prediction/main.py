# main.py
from src.utils import load_dataset
from src.data_preprocessing import preprocess_data
from src.model_training import train_model
from src.evaluation import evaluate_model

def main():
    # Load dataset
    df = load_dataset()

    # Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate
    results = evaluate_model(model, X_test, y_test)
    print("Model Evaluation Results:")
    print(results)

if __name__ == "__main__":
    main()
