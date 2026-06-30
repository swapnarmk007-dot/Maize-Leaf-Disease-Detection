"""
train.py
Train two models on the Maize / Corn leaf-disease dataset:
    1. A custom Convolutional Neural Network (CNN) built from scratch.
    2. VGG16 transfer learning (ImageNet weights, frozen convolutional base).

Dataset (download and unzip into ./Dataset so it looks like):
    Dataset/
        Blight/*.jpg
        Common_Rust/*.jpg
        Gray_Leaf_Spot/*.jpg
        Healthy/*.jpg

Sources:
    https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset
    https://github.com/R4j4n/Maize-Diseases-Detection

Usage:
    python train.py --model cnn   --epochs 20
    python train.py --model vgg   --epochs 15
    python train.py --model both  --epochs 20
"""

import argparse
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

from utils import IMG_SIZE, CLASS_NAMES

DATA_DIR = "Dataset"
MODEL_DIR = "models"
ART_DIR = "artifacts"
BATCH_SIZE = 32
SEED = 42

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ART_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def make_datasets(model_type):
    """Build train / validation tf.data pipelines with augmentation."""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_names=CLASS_NAMES,
        label_mode="categorical",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_names=CLASS_NAMES,
        label_mode="categorical",
    )

    augment = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    def prep(x, y, training):
        if training:
            x = augment(x)
        if model_type == "vgg":
            x = preprocess_input(x)        # mean-centering / BGR
        else:
            x = x / 255.0                  # simple scaling
        return x, y

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.map(lambda x, y: prep(x, y, True)).prefetch(AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: prep(x, y, False)).prefetch(AUTOTUNE)
    return train_ds, val_ds


# ----------------------------------------------------------------------
# Model definitions
# ----------------------------------------------------------------------
def build_cnn():
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(len(CLASS_NAMES), activation="softmax"),
    ], name="MaizeCNN")
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def build_vgg():
    base = VGG16(weights="imagenet", include_top=False,
                 input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False                      # freeze convolutional base
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(len(CLASS_NAMES), activation="softmax"),
    ], name="MaizeVGG16")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# ----------------------------------------------------------------------
# Train one model
# ----------------------------------------------------------------------
def train_one(model_type, epochs):
    print(f"\n=== Training {model_type.upper()} ===")
    train_ds, val_ds = make_datasets(model_type)
    model = build_cnn() if model_type == "cnn" else build_vgg()
    model.summary()

    ckpt = os.path.join(MODEL_DIR, f"{model_type}_model.h5")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt, save_best_only=True,
                                           monitor="val_accuracy"),
        tf.keras.callbacks.EarlyStopping(patience=5,
                                         restore_best_weights=True,
                                         monitor="val_accuracy"),
    ]

    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=epochs, callbacks=callbacks)

    model.save(ckpt)
    # Persist training curves for the web app
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(os.path.join(ART_DIR, f"{model_type}_history.json"), "w") as f:
        json.dump(hist, f, indent=2)

    print(f"Saved model -> {ckpt}")
    print(f"Saved history -> {ART_DIR}/{model_type}_history.json")
    return ckpt


# ----------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "vgg", "both"], default="both")
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()

    targets = ["cnn", "vgg"] if args.model == "both" else [args.model]
    for t in targets:
        train_one(t, args.epochs)

    print("\nTraining complete. Run `python evaluate.py` to generate "
          "ROC curves, confusion matrices and classification reports.")
