# model_training.py
from sklearn.naive_bayes import MultinomialNB

def train_model(X_train, y_train):
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    return clf
