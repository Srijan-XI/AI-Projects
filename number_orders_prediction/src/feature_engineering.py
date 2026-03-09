def create_time_features(df, date_col):
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['month'] = df[date_col].dt.month
    df['year'] = df[date_col].dt.year
    return df

def create_lag_features(df, target_col, lags=[1,7,14]):
    for lag in lags:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    return df

def create_moving_averages(df, target_col, windows=[7,14]):
    for window in windows:
        df[f'rolling_mean_{window}'] = df[target_col].rolling(window).mean()
    return df
