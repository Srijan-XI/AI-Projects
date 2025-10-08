# titanic_survival_prediction/src/data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def load_data(filepath):
    """Load Titanic dataset from a CSV file."""
    return pd.read_csv(filepath)

def clean_data(df):
    """Clean the Titanic dataset by handling missing values and dropping unnecessary columns."""
    # Drop columns that are not useful (if they exist)
    drop_cols = ['deck', 'embark_town', 'alive', 'class', 'who', 'adult_male']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    # Fill missing numerical values with median
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col].fillna(df[col].median(), inplace=True)

    # Fill missing categorical values with mode
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    return df

def encode_features(X):
    """Encode categorical features using OneHotEncoder."""
    encoder = OneHotEncoder(drop='first', sparse_output=False)  # Updated for sklearn>=1.2
    X_encoded = encoder.fit_transform(X.select_dtypes(include=['object']))
    X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(X.select_dtypes(include=['object']).columns))
    
    # Keep numerical columns
    X_num = X.select_dtypes(exclude=['object']).reset_index(drop=True)
    
    # Combine numerical and encoded categorical
    X_final = pd.concat([X_num, X_encoded_df], axis=1)
    return X_final, encoder
