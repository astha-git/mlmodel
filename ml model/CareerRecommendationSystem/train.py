import argparse
import os
import joblib
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

DEFAULT_MODEL_DIR = "/mnt/data/models"
os.makedirs(DEFAULT_MODEL_DIR, exist_ok=True)

def build_resource_map():
    # Minimal curated mapping: extend this with real URLs and entries.
    return {
        "AI/ML Engineer": [
            {"title": "Machine Learning Specialization - Coursera", "url": "https://www.coursera.org/specializations/machine-learning", "type": "course"},
            {"title": "Deep Learning Specialization - deeplearning.ai", "url": "https://www.coursera.org/specializations/deep-learning", "type": "cert"}
        ],
        "UI/UX Designer": [
            {"title": "Google UX Design Professional Certificate", "url": "https://coursera.org/professional-certificates/google-ux-design", "type": "cert"},
            {"title": "Interaction Design Foundation - Courses", "url": "https://www.interaction-design.org", "type": "course"}
        ],
        "DevOps Engineer": [
            {"title": "Google Cloud DevOps and SRE", "url": "https://cloud.google.com/training", "type": "course"},
            {"title": "Linux Foundation - DevOps", "url": "https://training.linuxfoundation.org", "type": "cert"}
        ],
        "Teacher / Lecturer": [
            {"title": "Foundations of Teaching for Learning (Coursera)", "url": "https://www.coursera.org", "type": "course"}
        ],
        "Research Scientist": [
            {"title": "Stanford CS231n (Deep Learning)", "url": "http://cs231n.stanford.edu", "type": "course"}
        ]
    }

def train(data_path, out_model):
    print("Loading data from:", data_path)
    df = pd.read_csv(data_path)
    if "career" not in df.columns:
        raise ValueError("Expected target column 'career' in dataset. Found columns: %s" % (df.columns.tolist(),))
    X = df.drop(columns=["career"])
    y = df["career"]

    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=42, stratify=y)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1))
    ])

    params = {
        "rf__max_depth": [10, None],
        "rf__min_samples_leaf": [1,2]
    }

    print("Starting GridSearchCV...")
    gs = GridSearchCV(pipe, params, cv=4, n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)

    print("Best params:", gs.best_params_)
    pred = gs.predict(X_test)
    print("Test accuracy: %.4f" % accuracy_score(y_test, pred))
    print(classification_report(y_test, pred))

    # Attach resources mapping
    artifact = {
        "model": gs.best_estimator_,
        "resource_map": build_resource_map()
    }

    joblib.dump(artifact, out_model, compress=3)
    print("Saved model artifact to:", out_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/mnt/data/career_dataset_500.csv")
    parser.add_argument("--out_model", type=str, default=os.path.join(DEFAULT_MODEL_DIR, "career_model.joblib"))
    args = parser.parse_args()
    train(args.data_path, args.out_model)
