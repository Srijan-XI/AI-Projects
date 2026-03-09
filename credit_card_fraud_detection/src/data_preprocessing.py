import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt
import seaborn as sns

class DataPreprocessor:
    """Class for preprocessing credit card fraud data"""
    
    def __init__(self):
        self.scaler = None
        self.pca = None
        self.feature_names = None
    
    def preprocess_data(self, df, test_size=0.2, random_state=42):
        """Main preprocessing pipeline"""
        print("Starting data preprocessing...")
        
        # Separate features and target
        X = df.drop(['Class'], axis=1)
        y = df['Class']
        self.feature_names = X.columns.tolist()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Scale the features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def scale_features(self, X_train, X_test, method='robust'):
        """Scale features using specified method"""
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Features scaled using {method} scaler")
        return X_train_scaled, X_test_scaled
    
    def handle_imbalance(self, X_train, y_train, method='smote'):
        """Handle class imbalance using various techniques"""
        print(f"Original training set distribution:")
        print(y_train.value_counts())
        
        if method == 'smote':
            sampler = SMOTE(random_state=42)
        elif method == 'undersample':
            sampler = RandomUnderSampler(random_state=42)
        elif method == 'smote_tomek':
            sampler = SMOTETomek(random_state=42)
        
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        
        print(f"After {method} resampling:")
        print(pd.Series(y_resampled).value_counts())
        
        return X_resampled, y_resampled
    
    def apply_pca(self, X_train, X_test, n_components=None, variance_threshold=0.95):
        """Apply PCA for dimensionality reduction"""
        if n_components is None:
            # Find number of components for desired variance
            pca_temp = PCA()
            pca_temp.fit(X_train)
            cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= variance_threshold) + 1
        
        self.pca = PCA(n_components=n_components, random_state=42)
        X_train_pca = self.pca.fit_transform(X_train)
        X_test_pca = self.pca.transform(X_test)
        
        explained_variance = sum(self.pca.explained_variance_ratio_)
        print(f"PCA applied: {n_components} components explain {explained_variance:.4f} of variance")
        
        return X_train_pca, X_test_pca
    
    def plot_pca_analysis(self, X, y, n_components=2):
        """Visualize data using PCA"""
        pca_viz = PCA(n_components=n_components)
        X_pca = pca_viz.fit_transform(X)
        
        plt.figure(figsize=(12, 5))
        
        # 2D PCA plot
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='RdYlBu', alpha=0.6)
        plt.colorbar(scatter)
        plt.xlabel(f'PC1 ({pca_viz.explained_variance_ratio_[0]:.3f} variance)')
        plt.ylabel(f'PC2 ({pca_viz.explained_variance_ratio_[1]:.3f} variance)')
        plt.title('PCA Visualization of Credit Card Transactions')
        
        # Explained variance plot
        plt.subplot(1, 2, 2)
        pca_full = PCA()
        pca_full.fit(X)
        cumsum = np.cumsum(pca_full.explained_variance_ratio_)
        plt.plot(range(1, len(cumsum) + 1), cumsum, 'bo-')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Explained Variance')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
