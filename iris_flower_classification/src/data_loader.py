#iris_flower_classification/src/data_loader.py
from sklearn.datasets import load_iris
import pandas as pd

def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df
