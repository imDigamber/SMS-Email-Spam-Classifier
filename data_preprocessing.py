# data_preprocessing.py
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    return text

def preprocess_data(df):
    df['text'] = df['v2'].apply(clean_text)
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = tfidf_vectorizer.fit_transform(df['text'])
    y = df['v1'].map({'ham': 0, 'spam': 1})
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
