import os
import joblib
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import make_scorer, f1_score


class ModelTrainer:
    """Class for training fraud detection models with speed optimizations."""

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.best_models: Dict[str, Any] = {}
        self.model_scores: Dict[str, Dict[str, Any]] = {}

    def initialize_models(self, scale_pos_weight=1, fast_dev_mode=True):
        """
        Initialize models.
        Args:
            scale_pos_weight (float): Used for XGBoost to handle imbalance.
            fast_dev_mode (bool): If True, skip slow models (SVM/XGBoost) for quick runs.
        """
        self.models = {
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=500,
                class_weight='balanced'
            ),
            'random_forest': RandomForestClassifier(
                random_state=42,
                class_weight='balanced',
                n_estimators=50  # Reduced for speed
            )
        }

        if not fast_dev_mode:
            self.models['xgboost'] = XGBClassifier(
                random_state=42,
                eval_metric='logloss',
                scale_pos_weight=scale_pos_weight,
                use_label_encoder=False,
                n_estimators=100,
                verbosity=0
            )
            self.models['svm'] = SVC(
                random_state=42,
                probability=True,
                class_weight='balanced'
            )

        print(f"Models initialized: {list(self.models.keys())}")

    def train_models(self, X_train, y_train, cv_folds=3, dev_sample_size=10000):
        """
        Train models with CV scoring on a smaller subset for speed.
        Args:
            cv_folds (int): Number of CV folds.
            dev_sample_size (int): Max samples to use for dev run.
        """
        # Use subset for quick dev runs
        if len(X_train) > dev_sample_size:
            X_train = X_train[:dev_sample_size]
            y_train = y_train[:dev_sample_size]
            print(f"⚡ Using subset of {dev_sample_size} samples for training (dev mode).")

        scorer = make_scorer(f1_score)
        for name, model in self.models.items():
            print(f"\n🔹 Training {name} ...")
            model.fit(X_train, y_train)
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring=scorer)
            self.model_scores[name] = {
                'mean_cv_score': cv_scores.mean(),
                'std_cv_score': cv_scores.std()
            }
            print(f"{name} - F1 CV Mean: {cv_scores.mean():.4f}")

    def hyperparameter_tuning(self, X_train, y_train, model_name='random_forest', cv_folds=2):
        """
        Tune hyperparameters with reduced CV folds for speed.
        """
        print(f"\n⚙️ Hyperparameter Tuning for {model_name} ...")
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100],
                'max_depth': [None, 10]
            },
            'logistic_regression': {
                'C': [0.1, 1],
                'solver': ['liblinear']
            }
        }

        if model_name not in param_grids:
            print(f"Skipping tuning: no param grid for {model_name}")
            return None

        model = self.models[model_name]
        grid = GridSearchCV(model, param_grids[model_name], cv=cv_folds, scoring='f1', n_jobs=-1, verbose=1)
        grid.fit(X_train[:5000], y_train[:5000])  # Limit data for speed
        self.best_models[model_name] = grid.best_estimator_
        print(f"✅ Best params: {grid.best_params_}")
        return grid.best_estimator_

    def save_models(self, save_dir='models/saved_models'):
        """Save all trained models."""
        os.makedirs(save_dir, exist_ok=True)
        for name, model in {**self.models, **self.best_models}.items():
            file_path = os.path.join(save_dir, f"{name}.pkl")
            joblib.dump(model, file_path)
            print(f"💾 Saved model: {file_path}")

    def load_models(self, save_dir='models/saved_models') -> Dict[str, Any]:
        """Load saved models."""
        loaded = {}
        for file in os.listdir(save_dir):
            if file.endswith('.pkl'):
                loaded[file[:-4]] = joblib.load(os.path.join(save_dir, file))
                print(f"📂 Loaded model: {file}")
        return loaded
