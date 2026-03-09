from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

def train_models(X, y):
    results = {}
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    results['LinearRegression'] = (lr, X_test, y_test)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    results['RandomForest'] = (rf, X_test, y_test)

    # XGBoost
    xgbr = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    xgbr.fit(X_train, y_train)
    results['XGBoost'] = (xgbr, X_test, y_test)

    return results
