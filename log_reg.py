import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import bz2
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from typing import Optional
import joblib

TRAIN_PATH = "../input/amazonreviews/train.ft.txt.bz2"
TEST_PATH = "../input/amazonreviews/test.ft.txt.bz2"


def read_decode_bz2(input_path):
    file = bz2.BZ2File(input_path)
    file_lines = file.readlines()
    file_lines = [x.decode("utf-8") for x in file_lines]
    return file_lines


def split_sub(file_lines):
    train_sentences = [x.split(" ", 1)[1][:-1] for x in file_lines]
    for i in range(len(train_sentences)):
        train_sentences[i] = re.sub("\d", "0", train_sentences[i])
    for i in range(len(train_sentences)):
        if (
            "www." in train_sentences[i]
            or "http:" in train_sentences[i]
            or "https:" in train_sentences[i]
            or ".com" in train_sentences[i]
        ):
            train_sentences[i] = re.sub(
                r"([^ ]+(?<=\.[a-z]{3}))", "<url>", train_sentences[i]
            )
    return train_sentences


def extract_label(file_lines):
    train_labels = [0 if x.split(" ")[0] == "__label__1" else 1 for x in file_lines]
    return train_labels


def file_to_sentences(input_path, test: Optional[bool] = None):
    data = read_decode_bz2(input_path)
    sentences = split_sub(data)
    if test is None:
        labels = extract_label(data)
        return sentences, labels
    return sentences


def create_dataframe(sentences, labels: Optional[list] = None):
    if labels is not None:
        df = pd.DataFrame({"sentences": sentences, "labels": labels})
        df["labels"] = df.labels.astype("category")
        return df
    else:
        df = pd.DataFrame({"sentences": sentences})
        return df


def prediction(model, X_train, y_train, X_valid, y_valid):
    model.fit(X_train, y_train)
    joblib.dump(model, "model.pkl")
    pred = model.predict(X_valid)
    acc = accuracy_score(y_valid, pred)
    f1 = f1_score(y_valid, pred)
    conf = confusion_matrix(y_valid, pred)
    return model, acc, f1, conf


def submission_csv(model, X_test):
    result = model.predict(X_test)
    df = pd.DataFrame({"prediction": result})
    df.to_csv("submission.csv", index=False)


def main():
    X, y = file_to_sentences(TRAIN_PATH)
    X_test = file_to_sentences(TEST_PATH, test=True)
    train_df = create_dataframe(X, y)
    test_df = create_dataframe(X_test)
    transformer = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), lowercase=True, max_features=20000
    )
    X = transformer.fit_transform(train_df["sentences"])
    X_test = transformer.transform(test_df["sentences"])
    y = train_df.labels
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression(C=1, random_state=42, n_jobs=-1)
    fit_model, acc, f1, conf = prediction(model, X_train, y_train, X_valid, y_valid)
    submission_csv(fit_model, X_test)
    print(f"Accuracy: {acc}")
    print(f"F1_Score: {f1}")
    print(f"Confusion Matrix: {conf}")


if __name__ == "__main__":
    main()
