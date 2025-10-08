import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from src.utils import FraudDetectionUtils
from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluator

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def main():
    """Main execution function"""
    print("=" * 60)
    print("CREDIT CARD FRAUD DETECTION PROJECT")
    print("=" * 60)
    print("Author: Srijan Kumar")
    print("Objective: Build ML models for fraud detection")
    print("=" * 60)

    # Initialize classes
    utils = FraudDetectionUtils()
    preprocessor = DataPreprocessor()
    trainer = ModelTrainer()
    evaluator = ModelEvaluator()

    # Step 1: Load Data
    print("\n1. LOADING DATA")
    print("-" * 30)

    # Load the dataset (download from Kaggle if not available)
    df = utils.load_data('data/creditcard.csv')

    if df is None:
        print("Please download the dataset from:")
        print("https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        return

    # Step 2: Exploratory Data Analysis
    print("\n2. EXPLORATORY DATA ANALYSIS")
    print("-" * 30)

    # Basic information
    utils.basic_info(df)

    # Visualizations
    utils.plot_class_distribution(df)

    # Feature analysis
    important_features = ['Time', 'Amount', 'V1', 'V2', 'V3', 'V4']
    utils.plot_feature_distributions(df, important_features)

    # Correlation analysis
    utils.plot_correlation_matrix(df)

    # Step 3: Data Preprocessing
    print("\n3. DATA PREPROCESSING")
    print("-" * 30)

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocessor.preprocess_data(df)

    # Handle class imbalance
    X_train_balanced, y_train_balanced = preprocessor.handle_imbalance(
        X_train, y_train, method='smote'
    )

    # PCA Visualization
    preprocessor.plot_pca_analysis(X_train, y_train)

    # Step 4: Model Training
    print("\n4. MODEL TRAINING")
    print("-" * 30)

    # Initialize and train models (with fast dev mode for quick testing)
    trainer.initialize_models(fast_dev_mode=True)  # Set to False for full training later
    trainer.train_models(X_train_balanced, y_train_balanced)

    # Hyperparameter tuning for Random Forest
    print("\nPerforming hyperparameter tuning...")
    best_rf = trainer.hyperparameter_tuning(
        X_train_balanced, y_train_balanced, 'random_forest'
    )

    # Step 5: Model Evaluation
    print("\n5. MODEL EVALUATION")
    print("-" * 30)

    # Evaluate all models
    models_to_evaluate = trainer.models.copy()
    if 'random_forest' in trainer.best_models:
        models_to_evaluate['random_forest_tuned'] = trainer.best_models['random_forest']

    for name, model in models_to_evaluate.items():
        evaluator.evaluate_model(model, X_test, y_test, name)

    # Create comparison table
    comparison_df = evaluator.create_model_comparison_table()

    # Plot ROC curves
    evaluator.plot_roc_curve(X_test, y_test, models_to_evaluate)

    # Plot Precision-Recall curves
    evaluator.plot_precision_recall_curve(X_test, y_test, models_to_evaluate)

    # Plot confusion matrices for top models (only if evaluated)
    top_models = ['random_forest', 'xgboost']
    for model_name in top_models:
        if model_name in evaluator.evaluation_results:
            evaluator.plot_confusion_matrix(model_name)

    # Feature importance analysis (if RF exists)
    if 'random_forest' in trainer.models:
        evaluator.plot_feature_importance(
            trainer.models['random_forest'],
            preprocessor.feature_names,
            'Random Forest'
        )

    # Step 6: Save Models
    print("\n6. SAVING MODELS")
    print("-" * 30)
    trainer.save_models()

    # Step 7: Summary and Recommendations
    print("\n7. PROJECT SUMMARY")
    print("-" * 30)

    if comparison_df is not None and not comparison_df.empty:
        best_model_idx = comparison_df['F1 Score'].idxmax()
        best_model = comparison_df.iloc[best_model_idx]

        print(f"Best performing model: {best_model['Model']}")
        print(f"F1 Score: {best_model['F1 Score']:.4f}")
        print(f"ROC-AUC: {best_model['ROC-AUC']:.4f}")
        print(f"Precision: {best_model['Precision (Class 1)']:.4f}")
        print(f"Recall: {best_model['Recall (Class 1)']:.4f}")

    print("\n" + "=" * 60)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("Models saved in 'models/saved_models/' directory")
    print("=" * 60)


if __name__ == "__main__":
    main()
