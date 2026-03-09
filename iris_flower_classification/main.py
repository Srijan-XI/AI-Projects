# iris_flower_classification/main.py
from src.data_loader import load_data
from src.data_visualization import plot_pairwise, plot_correlation
from src.model_training import split_data, train_model
from src.model_evaluation import evaluate_model

def main():
    print("📥 Loading dataset...")
    df = load_data()
    print(df.head())

    print("\n📊 Visualizing dataset...")
    plot_pairwise(df)
    plot_correlation(df)

    print("\n✂ Splitting dataset...")
    X_train, X_test, y_train, y_test = split_data(df)

    print("\n🤖 Training model...")
    model = train_model(X_train, y_train, model_type="logistic")

    print("\n📈 Evaluating model...")
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
