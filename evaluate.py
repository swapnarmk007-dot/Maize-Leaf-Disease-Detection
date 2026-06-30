"""
evaluate.py
Evaluate trained model(s) on a held-out validation split and persist:
    - <model>_metrics.json   : overall accuracy + per-class precision/recall/f1
    - <model>_report.json    : full sklearn classification_report (dict)
    - <model>_confusion.json : confusion-matrix counts
    - <model>_roc.json       : per-class fpr / tpr / auc for ROC plotting

These JSON artifacts are read by app.py and rendered with the purple theme,
so the deployed web app needs no test data and no TensorFlow at view time.

Usage:
    python evaluate.py --model cnn
    python evaluate.py --model vgg
    python evaluate.py --model both
"""

import argparse
import json
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc)
from sklearn.preprocessing import label_binarize
from tensorflow.keras.applications.vgg16 import preprocess_input

from utils import IMG_SIZE, CLASS_NAMES

DATA_DIR = "Dataset"
MODEL_DIR = "models"
ART_DIR = "artifacts"
BATCH_SIZE = 32
SEED = 42

os.makedirs(ART_DIR, exist_ok=True)


def get_val_dataset(model_type):
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_names=CLASS_NAMES,
        label_mode="categorical",
        shuffle=False,
    )

    def prep(x, y):
        x = preprocess_input(x) if model_type == "vgg" else x / 255.0
        return x, y

    return val_ds.map(prep)


def evaluate_one(model_type):
    path = os.path.join(MODEL_DIR, f"{model_type}_model.h5")
    if not os.path.exists(path):
        print(f"[skip] {path} not found - train the model first.")
        return

    print(f"\n=== Evaluating {model_type.upper()} ===")
    model = tf.keras.models.load_model(path)
    val_ds = get_val_dataset(model_type)

    y_true, y_prob = [], []
    for x, y in val_ds:
        y_prob.append(model.predict(x, verbose=0))
        y_true.append(y.numpy())
    y_prob = np.concatenate(y_prob)
    y_true = np.concatenate(y_true)

    y_true_idx = np.argmax(y_true, axis=1)
    y_pred_idx = np.argmax(y_prob, axis=1)

    # ---- metrics + classification report -----------------------------
    report = classification_report(
        y_true_idx, y_pred_idx, target_names=CLASS_NAMES,
        output_dict=True, zero_division=0)
    accuracy = float((y_true_idx == y_pred_idx).mean())

    with open(os.path.join(ART_DIR, f"{model_type}_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(ART_DIR, f"{model_type}_metrics.json"), "w") as f:
        json.dump({"accuracy": accuracy,
                   "n_samples": int(len(y_true_idx))}, f, indent=2)

    # ---- confusion matrix --------------------------------------------
    cm = confusion_matrix(y_true_idx, y_pred_idx).tolist()
    with open(os.path.join(ART_DIR, f"{model_type}_confusion.json"), "w") as f:
        json.dump({"matrix": cm, "labels": CLASS_NAMES}, f, indent=2)

    # ---- ROC (one-vs-rest, per class) --------------------------------
    y_bin = label_binarize(y_true_idx, classes=list(range(len(CLASS_NAMES))))
    roc = {}
    for i, name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
                     "auc": float(auc(fpr, tpr))}
    with open(os.path.join(ART_DIR, f"{model_type}_roc.json"), "w") as f:
        json.dump(roc, f, indent=2)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Artifacts written to ./{ART_DIR}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "vgg", "both"], default="both")
    args = ap.parse_args()
    targets = ["cnn", "vgg"] if args.model == "both" else [args.model]
    for t in targets:
        evaluate_one(t)
