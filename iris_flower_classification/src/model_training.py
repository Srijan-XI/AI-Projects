#iris_flower_classification/src/model_training.py
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

def split_data(df):
    X = df.drop(columns=['species'])
    y = df['species']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train, model_type="logistic"):
    if model_type == "logistic":
        model = LogisticRegression(max_iter=200)
    elif model_type == "knn":
        model = KNeighborsClassifier(n_neighbors=5)
    else:
        raise ValueError("Unsupported model type. Use 'logistic' or 'knn'.")
    
    model.fit(X_train, y_train)
    return model
