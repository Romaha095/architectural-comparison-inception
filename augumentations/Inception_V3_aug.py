# -*- coding: utf-8 -*-


#!pip install opendatasets

import opendatasets as od

od.download("https://www.kaggle.com/datasets/kasikrit/idc-dataset")

# Knižnice
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

#cesta k datasetu
#/content/idc-dataset/IDC


# Načítanie datasetu
DATASET = "idc-dataset/IDC"

TRAINING_DIR = os.path.join(DATASET, "training")
VALIDATION_DIR = os.path.join(DATASET, "validation")
TESTING_DIR = os.path.join(DATASET, "testing")


# Nastavenie veľkosti obrázkov a počet kanálov
IMG_SIZE = 75
COLOR_CHANNELS = 3
BATCH_SIZE = 128

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.2
)

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

train_gen = train_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='sparse'
)

val_gen = datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='sparse'
)

test_gen = datagen.flow_from_directory(
    TESTING_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    shuffle=False
)

NUM_CLASSES = train_gen.num_classes



base_model = InceptionV3(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze base model
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
outputs = Dense(2, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=outputs)



early_stopping = EarlyStopping(
    monitor="val_loss",      # best choice for imbalance
    patience=3,              # stop after 3 epochs with no improvement
    restore_best_weights=True,
    verbose=1
)

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

# Unfreeze top layers of InceptionV3
for layer in base_model.layers[-50:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=0.00045),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_finetune = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    callbacks=[early_stopping]
)


# Evaluate
scores = model.evaluate(test_gen, verbose=0)
print(f"Evaluating Model — Loss: {scores[0]:.4f}, Accuracy: {scores[1]:.4f}")

# Evaluate on TEST SET
y_true = []
y_pred = []

test_gen.reset()  # VERY IMPORTANT

# Predict probabilities
y_prob = model.predict(test_gen, verbose=0)

# Convert softmax → class index
y_pred = np.argmax(y_prob, axis=1)

# True labels
y_true = test_gen.classes


class_names = ["0", "1"]

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
))


cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix:\n", cm)


bal_acc = balanced_accuracy_score(y_true, y_pred)
print(f"\nBalanced Accuracy: {bal_acc:.4f}")

