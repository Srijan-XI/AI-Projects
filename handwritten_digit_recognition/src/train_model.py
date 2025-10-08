#handwritten_digit_recognition/src/train_model.py
from sklearn.ensemble import RandomForestClassifier

def train_classifier(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf
