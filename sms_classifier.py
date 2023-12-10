# sms_classifier.py
import pandas as pd
from data_preprocessing import clean_text, preprocess_data
from model_training import train_model
from model_evaluation import evaluate_model



# Load the dataset
df = pd.read_csv('spam.csv', encoding='latin-1')

# Preprocess the data
X_train, X_test, y_train, y_test = preprocess_data(df)

# Train the model
clf = train_model(X_train, y_train)

# Evaluate the model
evaluate_model(clf, X_test, y_test)
