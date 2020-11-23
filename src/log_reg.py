import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib

TRAIN_PATH = "../input/amazontrainreviews/train.csv"


def prediction(model, X_train, y_train, X_valid, y_valid):
    model.fit(X_train, y_train)
    pred = model.predict(X_valid)
    acc = accuracy_score(y_valid, pred)
    f1 = f1_score(y_valid, pred)
    conf = confusion_matrix(y_valid, pred)
    joblib.dump(model, f"model_acc_{acc:.5f}.pkl")
    return model, acc, f1, conf


def main() -> None:
    train_val = pd.read_csv(TRAIN_PATH, index_col=0)
    train_val.reset_index(drop=True, inplace=True)
    transformer = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 3), lowercase=True, max_features=100000
    )
    X = transformer.fit_transform(train_val["sentences"])
    y = train_val.labels
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression(C=1, random_state=42, n_jobs=-1)
    fit_model, acc, f1, conf = prediction(model, X_train, y_train, X_valid, y_valid)
    print(f"Accuracy: {acc:.5f}")
    print(f"F1_Score: {f1:.5f}")
    print(f"Confusion Matrix: {conf}")


if __name__ == "__main__":
    main()
