#handwritten_digit_recognition/src/load_data.py
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def load_and_preprocess_data():
    # Load digits dataset
    digits = load_digits()
    X_images = digits.images   # Shape: (n_samples, 8, 8)
    y = digits.target

    # Normalize pixel values (0–16 -> 0–1)
    X_images = X_images / 16.0

    # Flatten images for classifier
    n_samples = len(X_images)
    X_flat = X_images.reshape((n_samples, -1))

    # Train-test split (keep images for visualization)
    X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(
        X_flat, y, X_images, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, images_test
