import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionUtils:
    """Utility class for fraud detection project"""
    
    @staticmethod
    def load_data(filepath):
        """Load the credit card dataset"""
        try:
            df = pd.read_csv(filepath)
            print(f"Dataset loaded successfully: {df.shape}")
            return df
        except FileNotFoundError:
            print(f"File not found: {filepath}")
            print("Please download the dataset from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
            return None
    
    @staticmethod
    def basic_info(df):
        """Display basic information about the dataset"""
        print("Dataset Information:")
        print("=" * 50)
        print(f"Shape: {df.shape}")
        print(f"Memory Usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
        print("\nData Types:")
        print(df.dtypes.value_counts())
        print("\nMissing Values:")
        print(df.isnull().sum().sum())
        print("\nClass Distribution:")
        print(df['Class'].value_counts())
        fraud_percentage = (df['Class'].sum() / len(df)) * 100
        print(f"Fraud Percentage: {fraud_percentage:.4f}%")
    
    @staticmethod
    def plot_class_distribution(df):
        """Plot class distribution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Count plot
        df['Class'].value_counts().plot(kind='bar', ax=ax1, color=['skyblue', 'salmon'])
        ax1.set_title('Class Distribution')
        ax1.set_xlabel('Class (0: Normal, 1: Fraud)')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=0)
        
        # Pie chart
        df['Class'].value_counts().plot(kind='pie', ax=ax2, autopct='%1.4f%%', 
                                       colors=['skyblue', 'salmon'])
        ax2.set_title('Class Distribution (Percentage)')
        ax2.set_ylabel('')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_feature_distributions(df, features, target='Class'):
        """Plot feature distributions by class"""
        n_features = len(features)
        n_rows = (n_features + 2) // 3
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(18, n_rows * 5))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, feature in enumerate(features):
            if i < len(axes):
                for class_val in df[target].unique():
                    data = df[df[target] == class_val][feature]
                    axes[i].hist(data, alpha=0.7, label=f'Class {class_val}', bins=30)
                axes[i].set_title(f'{feature} Distribution')
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Frequency')
                axes[i].legend()
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].hide()
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_correlation_matrix(df, figsize=(12, 10)):
        """Plot correlation matrix"""
        plt.figure(figsize=figsize)
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
        
        # Show top correlations with target
        target_corr = correlation_matrix['Class'].abs().sort_values(ascending=False)
        print("\nTop 10 features correlated with fraud:")
        print(target_corr[1:11])  # Exclude self-correlation
