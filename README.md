
# AI Projects Repository
This repository contains a collection of AI and machine learning projects, each demonstrating different techniques and applications. Each project is self-contained with its own data, code, and instructions.

## Project List

### 1. Air Quality Analysis & Forecasting
**Folder:** `air-quality-analysis-forecasting/`
- Multiple notebooks for AQI analysis and forecasting using SARIMA, RNN-LSTM, and Facebook Prophet.
- Dataset included in `air-quality-dataset/`.
- **How to run:** Open any notebook (`*.ipynb`) in Jupyter and follow the cells.

### 2. Chatbot
**Folder:** `chatbot/`
- AI-powered chatbot using deep learning (Keras, NLTK, Flask web app).
- Files: `app.py`, `chatbot_model.py`, `intents.json`, and pre-trained model files.
- **How to run:**
	1. Install dependencies (Flask, TensorFlow, NLTK).
	2. Run `python app.py` and open the web interface.

### 3. Credit Card Fraud Detection
**Folder:** `credit_card_fraud_detection/`
- End-to-end ML pipeline: preprocessing, model training, evaluation, and visualization.
- Uses Logistic Regression, Random Forest, XGBoost, and more.
- Data: `data/creditcard.csv` (download from Kaggle if missing).
- **How to run:**
	1. Install dependencies from `requirements.txt`.
	2. Run `python main.py`.
	3. Explore results in `results/` and saved models in `models/`.

### 4. Movie Recommendation System
**Folder:** `movie_recommender/`
- Content-based recommender using TF-IDF on movie metadata.
- Data: `data/movies.csv`.
- **How to run:**
	1. Install dependencies (see `requirements.txt`).
	2. Run `python src/main.py` and enter a movie title for recommendations.

### 5. Number Orders Prediction
**Folder:** `number_orders_prediction/`
- Time series forecasting for predicting next values in a sequence.
- Includes feature engineering, model training, and evaluation.
- Notebooks for EDA in `notebooks/`.
- **How to run:**
	1. Install dependencies from `requirements.txt`.
	2. Run `python main.py`.

### 6. Titanic Survival Prediction
**Folder:** `titanic_survival_prediction/`
- Predicts passenger survival using classification models.
- Data: `data/titanic.csv`.
- **How to run:**
	1. Install dependencies from `requirements.txt`.
	2. Run `python main.py`.

### 7. Iris Flower Classification
**Folder:** `iris_flower_classification/`
- Classifies iris species using classic ML algorithms.
- Includes data loading, visualization, training, and evaluation.
- **How to run:**
	1. (Optional) Review `folder_structure.txt` for details.
	2. Run `python main.py`.

### 8. Handwritten Digit Recognition
**Folder:** `handwritten_digit_recognition/`
- Recognizes handwritten digits using scikit-learn and classic ML models.
- **How to run:**
	1. Install dependencies from `requirements.txt`.
	2. Run `python main.py`.

### 9. House Price Prediction
**Folder:** `house_price_prediction/`
- Regression model to predict house prices based on features.
- Data: `data/housing.csv`.
- **How to run:**
	1. Run `python main.py`.

---

## General Getting Started
1. Clone the repository and navigate to the desired project folder:
	 ```bash
	 git clone <repository-url>
	 cd <project-folder>
	 ```
2. Install dependencies:
	 ```bash
	 pip install -r requirements.txt
	 ```
3. Run the main script or open the notebook as described above.

## Reports
PDF reports for some projects are available in the `Z report/` folder.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License
This repository is licensed under the MIT License. See the LICENSE file for more details.# AI-Projects-Repository

