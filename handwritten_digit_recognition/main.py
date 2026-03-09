#handwritten_digit_recognition/main.py
from src.load_data import load_and_preprocess_data
from src.train_model import train_classifier
from src.evaluate_model import evaluate_classifier
from src.visualize_results import visualize_predictions

def main():
    # 1. Load and preprocess data
    X_train, X_test, y_train, y_test, images_test = load_and_preprocess_data()

    # 2. Train model
    model = train_classifier(X_train, y_train)

    # 3. Evaluate model
    accuracy, predictions = evaluate_classifier(model, X_test, y_test)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # 4. Visualize predictions
    visualize_predictions(images_test, predictions, y_test)

if __name__ == "__main__":
    main()
