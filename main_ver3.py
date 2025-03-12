import random
import numpy as np
import pandas as pd
import albumentations as A
from sklearn.model_selection import GridSearchCV, train_test_split
import pickle
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import os
from mnist import MNIST

MODELS = {
    "svm": {
        "file": "models/svm_model.pkl",
        "class": svm.SVC(),
        "params": {'kernel': ['rbf'], 'gamma': ['scale', 0.001, 0.01, 0.1], 'C': [0.1, 1, 10, 100]}
    },
    "gaussian_naive_bayes": {
        "file": "models/gaussian_naive_bayes_model.pkl",
        "class": GaussianNB(),
        "params": {}
    },
    "decision_tree": {
        "file": "models/decision_tree_model.pkl",
        "class": tree.DecisionTreeClassifier(),
        "params": {'max_depth': [5, 10, 15], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]}
    },
    "random_forest": {
        "file": "models/random_forest_model.pkl",
        "class": RandomForestClassifier(),
        "params": {'n_estimators': [50, 100], 'max_depth': [5, 10], 'min_samples_split': [2, 5]}
    },
    "k_nearest_neighbors": {
        "file": "models/k_nearest_neighbors_model.pkl",
        "class": KNeighborsClassifier(),
        "params": {'n_neighbors': [3, 5, 7], 'metric': ['euclidean', 'manhattan']}
    },
    "sgd": {
        "file": "models/sgd_model.pkl",
        "class": SGDClassifier(),
        "params": {'loss': ['hinge', 'log_loss'], 'penalty': ['l2', 'l1'], 'max_iter': [100, 200]}
    },
    "xgboost": {
        "file": "models/xgboost_model.pkl",
        "class": XGBClassifier(),
        "params": {'max_depth': [3, 5], 'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1]}
    },
}


def augment(image):
    if image.dtype != np.uint8:  # Ensure the image is of type uint8
        image = (image * 255).astype(np.uint8)

    transform = A.Compose([
        A.Affine(scale=(0.9, 1.1), translate_percent=(0.1, 0.1), rotate=(-15, 15), shear=(-10, 10), p=0.7),
        A.GaussianBlur(blur_limit=(3, 5), p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GridDistortion(p=0.3),
        A.ElasticTransform(p=0.3),
    ])

    transformed = transform(image=image)
    return transformed['image']


def preprocess_image(image):
    if image.shape != (28, 28):
        raise ValueError("Image must be of size 28x28")

    image = augment(image)
    image = 255 - image  # Invert image
    image = image / 255.0  # Normalize to [0, 1]
    return image.flatten()


def load_mnist_data():
    mndata = MNIST("MNIST")
    images, labels = mndata.load_training()
    processed_images = [preprocess_image(np.array(image).reshape(28, 28)) for image in images]
    return pd.DataFrame(processed_images), pd.Series(labels)


def train_and_evaluate_model(model_name, model, X_train, y_train, X_val, y_val, param_grid=None):
    if param_grid:
        print(f"Tuning hyperparameters for {model_name}...")
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        print(f"Best hyperparameters for {model_name}: {grid_search.best_params_}")

    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    val_accuracy = accuracy_score(y_val, model.predict(X_val))

    print(f"{model_name} Training Accuracy: {train_accuracy * 100:.2f}%")
    print(f"{model_name} Validation Accuracy: {val_accuracy * 100:.2f}%")

    return model


def train_and_save_models(X_train, y_train, X_val, y_val):
    trained_models = {}
    for model_name, model_data in MODELS.items():
        model = train_and_evaluate_model(
            model_name,
            model_data['class'],
            X_train, y_train, X_val, y_val,
            model_data['params'] if model_data['params'] else None
        )
        trained_models[model_name] = model
        with open(model_data['file'], 'wb') as f:
            pickle.dump(model, f)
        print(f"{model_name} model saved successfully.")
    return trained_models


def prepare_data():
    print("Loading and preprocessing MNIST data...")
    X, y = load_mnist_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    return X_train, X_val, y_train, y_val


def run_pipeline():
    X_train, X_val, y_train, y_val = prepare_data()
    train_and_save_models(X_train, y_train, X_val, y_val)


if __name__ == "__main__":
    run_pipeline()
