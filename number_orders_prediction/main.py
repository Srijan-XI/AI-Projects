import os
from src.data_preprocessing import load_data, preprocess_data
from src.feature_engineering import create_time_features, create_lag_features, create_moving_averages
from src.model_training import train_models
from src.evaluation import evaluate_model, plot_predictions
import joblib

# Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "train.csv")
TARGET_COL = "sales"
DATE_COL = "date"
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load and preprocess
df = load_data(DATA_PATH)
df, _ = preprocess_data(df)

# Feature engineering - use minimal lags/windows
df = create_time_features(df, DATE_COL)
df = create_lag_features(df, TARGET_COL, lags=[1])          # only 1-day lag
df = create_moving_averages(df, TARGET_COL, windows=[2])    # 2-day rolling mean

# Drop only rows with NaNs in lag and rolling columns
needed_cols = [TARGET_COL, 'lag_1', 'rolling_mean_2']
df = df.dropna(subset=needed_cols)

print(f"Data shape after dropping NaNs: {df.shape}")
if df.shape[0] == 0:
    raise ValueError("No data left after dropping NaNs. Check your lag/window sizes or dataset size.")

# Prepare features & target
X = df.drop(columns=[TARGET_COL, DATE_COL])
y = df[TARGET_COL]

# Train models
results = train_models(X, y)

# Evaluate & save best
best_model = None
best_rmse = float("inf")

for name, (model, X_test, y_test) in results.items():
    metrics, preds = evaluate_model(model, X_test, y_test)
    print(f"{name} Performance: {metrics}")
    if metrics["RMSE"] < best_rmse:
        best_rmse = metrics["RMSE"]
        best_model = model
        plot_predictions(y_test, preds, title=f"{name}: Actual vs Predicted")

os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(best_model, os.path.join(MODEL_DIR, "trained_model.pkl"))
print(f"✅ Best model saved with RMSE: {best_rmse}")
