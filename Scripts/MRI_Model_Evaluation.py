# -*- coding: utf-8 -*-

#knižnice
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from sklearn.metrics import (balanced_accuracy_score, classification_report, confusion_matrix)


#Názov skúšaného modelu
MODEL_NAME = "MRI_Inception_frozen.h5"

#Nastavenie veľkosti obrázkov a počet kanálov
IMAGE_SIZE = 299
COLOR_CHANNELS = 1
BATCH_SIZE = 16


#spracuvanie a delenie datasetu
DATASET_PATH = "mri_data"
training_data = []
CATEGORIES = ["glioma","meningioma","notumor","pituitary"]

for category in CATEGORIES:
    path = os.path.join(DATASET_PATH, category) #cesta k obrázkom -> mel_spectrograms_images/pop
    class_number = CATEGORIES.index(category) #0-> blues, 1-> classical ... 9-> rock
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        new_img = cv2.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE))
        #training_data.append([img_array, class_number])
        training_data.append([new_img, class_number])

X = []
Y = []
for features, labels in training_data:
    X.append(features)
    Y.append(labels)

X = np.array(X).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
Y = np.array(Y)


#train / val / test split 70:15:15
x_train, x_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=0.30)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, stratify=y_test, test_size=0.50)


model = load_model(MODEL_NAME)


# Evaluate
scores = model.evaluate(x_test, y_test, verbose=0)
print(f"Evaluating Model — Loss: {scores[0]:.4f}, Accuracy: {scores[1]:.4f}")


# Evaluate on TEST SET
y_true = []
y_pred = []

# Predict probabilities
y_prob = model.predict(x_test, verbose=0)

# Convert softmax → class index
y_pred = np.argmax(y_prob, axis=1)

# True labels
y_true = y_test

class_names = ["Glioma", "Meningioma", "No-tumor" ,"Pituitary"]

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


