import random
import numpy as np
import pandas as pd
from PIL import Image
import glob
import os
from sklearn.model_selection import GridSearchCV, train_test_split
import seaborn as sns
import pickle
from mnist import MNIST
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score #for loss calculation
import albumentations as A #for augmentation
import cv2

sns.set(rc={'figure.figsize': (11.7, 8.27)})
palette = sns.color_palette("bright", 28)

# Global models dictionary
MODELS = {
    "svm": {
        "file": "models/svm_model.pkl",
        "class": svm.SVC(),
        "params": {'kernel': ['rbf'], 'gamma': ['scale', 0.001, 0.01, 0.1], 'C': [0.1, 1, 10, 100]} #added parameters
    },
    "gaussian_naive_bayes": {
        "file": "models/gaussian_naive_bayes_model.pkl",
        "class": GaussianNB(),
        "params": {} #no parameters
    },
    "decision_tree": {
        "file": "models/decision_tree_model.pkl",
        "class": tree.DecisionTreeClassifier(),
        "params": {'max_depth': [5,10,15,20,25], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]} #added parameters
    },
    "random_forest": {
        "file": "models/random_forest_model.pkl",
        "class": RandomForestClassifier(),
        "params": {'max_depth': [5,10,15,20,25], 'n_estimators':[50,100,150,200],'min_samples_split': [2, 5, 10],'min_samples_leaf': [1, 2, 4]} #added parameters
    },
    "k_nearest_neighbors": {
        "file": "models/k_nearest_neighbors_model.pkl",
        "class": KNeighborsClassifier(),
        "params": {'n_neighbors': [3, 5, 7, 9,11], 'metric': ['euclidean', 'manhattan', 'minkowski']} #added parameters
    },
    "stochastic_gradient_decent": {
        "file": "models/stochastic_gradient_decent_model.pkl",
        "class": SGDClassifier(),
        "params": {'loss':['hinge', 'log_loss', 'modified_huber'], 'penalty':['l2', 'l1', 'elasticnet'], 'max_iter': [50, 100, 150, 200]} #added parameters
    },
    "xgboost": {
        "file": "models/xgboost_model.pkl",
        "class": XGBClassifier(),
        "params": {'max_depth': [3, 4, 5, 6, 7], 'n_estimators': [50, 100, 150, 200], 'learning_rate': [0.01, 0.05, 0.1, 0.2]} #added parameters
    },
}


# Augmentation
def augment(image):
    transform = A.Compose([
        A.Rotate(limit=10, p=0.5), #rotate
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5), # shift, rotate, scale
        A.GaussianBlur(blur_limit=(3,7), p=0.5), #blur
        A.RandomBrightnessContrast(p=0.5), #brightness contrast
        A.Affine(shear={'x': (-10, 10), 'y': (-10, 10)}, p=0.5) #skew
    ])

    transformed = transform(image=image)
    transformed_image = transformed['image']
    return transformed_image

def import_MNIST_dataset():
    """Imports the MNIST dataset.

    Returns:
        tuple: A tuple containing (images, labels).
               images: A list of flattened image arrays.
               labels: A list of corresponding labels.
    """
    mndata = MNIST("datasets/training_data/MNIST")
    images, labels = mndata.load_training()

    # index = random.randrange(0, len(images))
    # print(mndata.display(images[index]))
    return images, labels


def preprocess_mnist_image(image):
    """Preprocesses a single MNIST image (28x28).

    Args:
        image (list): A flattened 28x28 image array.

    Returns:
        numpy.ndarray: Preprocessed image as a flattened 28x28 array.
    """
    img = np.array(image).reshape(28, 28)
    img = augment(img) #augmented image
    img = 255 - img
    img = img / 255.0
    return img.ravel()


def preprocess_single_image(image_path):
    """Preprocesses a single digit image.

    Args:
        image_path (str): Path to the image.

    Returns:
        numpy.ndarray: Preprocessed image as a flattened array, or None if error.
    """
    try:
        img = Image.open(image_path)
        img = img.resize((28, 28))
        img = img.convert('L')
        img = np.array(img)
        img = 255 - img
        img = img / 255.0
        img = img.ravel()
        return img
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None


def preprocess_dataset(data, target, save_path=None, purpose="training"):
    """Preprocesses a dataset of images.

    Args:
        data (list): The image data to preprocess. Either a path or an imported MNIST data.
        target (list): the target labels.
        save_path (str, optional): Path to save the preprocessed DataFrame to a CSV. Defaults to None.
        purpose (str, optional): Specifies the purpose of preprocessing. Either "training" or "testing". Defaults to "training".

    Returns:
        pandas.DataFrame: DataFrame containing the preprocessed image data and target labels.
    """
    processed_data = []
    processed_target = []

    if purpose == "training":
        print("Processing MNIST data")
        for i, image in enumerate(data):
            processed_img = preprocess_mnist_image(image)
            if processed_img is not None:
                processed_data.append(processed_img.tolist())
                processed_target.append(target[i])
                if i % 10000 == 0:
                    print(f"Processed MNIST image: {i}")

    elif purpose == "testing":
        print("Processing Test data")
        for entry in glob.iglob(data):
            print(f"Processing class: {entry}")
            for image_path in glob.iglob(entry + '/*.jpg'):
                processed_img = preprocess_single_image(image_path)
                if processed_img is not None:
                    processed_data.append(processed_img.tolist())
                    processed_target.append(os.path.basename(entry))
                    print(f"Processed image: {image_path}")

    else:
        raise ValueError("Invalid purpose. Must be either 'training' or 'testing'.")
    
    if not processed_data:
        print("No images were processed.")
        return pd.DataFrame()

    df = pd.DataFrame(processed_data)
    df["target"] = processed_target
    df = shuffle(df).reset_index(drop=True)

    if save_path:
        csv_filename = f"{save_path}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"Preprocessed {purpose} data saved to {csv_filename}")

    return df


def train_and_save_models_GRID(X_train, y_train):
    """Trains multiple classification models and saves them to disk.

    Args:
        X_train (pandas.DataFrame): Training data.
        y_train (pandas.Series): Training labels.

    Returns:
        dict: A dictionary of trained models.
    """

    trained_models = {}
    for model_name, model_data in MODELS.items():
        print(f"Training {model_name}...")
        model = model_data["class"]
        
        param_grid = model_data["params"]
        # Perform Grid Search
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1) #added gridsearch
        grid_search.fit(X_train.values, y_train.values)

        # Select the best model from grid search
        best_model = grid_search.best_estimator_

        # Retrain on the whole data
        best_model.fit(X_train.values, y_train.values)
        
        trained_models[model_name] = best_model
        with open(model_data["file"], "wb") as f:
            pickle.dump(best_model, f)
        print(f"Saved {model_name} model")

    return trained_models

def train_and_save_models(X_train, y_train):
    """Trains multiple classification models and saves them to disk.

    Args:
        X_train (pandas.DataFrame): Training data.
        y_train (pandas.Series): Training labels.

    Returns:
        dict: A dictionary of trained models.
    """

    trained_models = {}
    for model_name, model_data in MODELS.items():
        print(f"Training {model_name}...")
        model = model_data["class"]
        model.fit(X_train.values, y_train.values)
        trained_models[model_name] = model
        with open(model_data["file"], "wb") as f:
            pickle.dump(model, f)
        print(f"Saved {model_name} model")

    return trained_models


def load_trained_models():
    """Loads trained models from disk.

    Returns:
        dict: A dictionary of loaded models, or None if no models are found.
    """
    loaded_models = {}
    for model_name, model_data in MODELS.items():
        try:
            with open(model_data["file"], "rb") as f:
                loaded_models[model_name] = pickle.load(f)
                print(f"Loaded {model_name} model")
        except FileNotFoundError:
            print(f"{model_name} model file not found.")
    if not loaded_models:
        return None
    return loaded_models


def predict_and_evaluate(X_test, y_test, models):
    """Predicts using the loaded models and evaluates their accuracy.

    Args:
        X_test (pandas.DataFrame): Test data.
        y_test (pandas.Series): Test labels.
        models (dict): Dictionary of loaded models.

    Returns:
        None
    """
    if models is None:
        print("No models were loaded. Cannot predict.")
        return

    for model_name, model in models.items():
        predicted = model.predict(X_test.values)
        accuracy = (len(X_test[predicted == y_test]) / len(X_test)) * 100
        print(f"{model_name.replace('_', ' ').title()} Accuracy: {accuracy}")


def predict_single_digit(image_path, models):
    """Predicts the digit in a single image using the loaded models.

    Args:
        image_path (str): Path to the image.
        models (dict): Dictionary of loaded models.

    Returns:
        dict: Predictions from each model, or None if error.
    """
    preprocessed_image = preprocess_single_image(image_path)
    if preprocessed_image is None:
        return None

    predictions = {}
    for model_name, model in models.items():
        prediction = model.predict([preprocessed_image])  # Pass as a list of one element
        predictions[model_name] = prediction[0]
        print(f"{model_name.replace('_', ' ').title()} prediction: {prediction[0]}")
    return predictions


def load_preprocessed_data(csv_path):
    """Loads preprocessed data from a CSV file.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        pandas.DataFrame: The loaded data, or None if the file is not found.
    """
    try:
        data = pd.read_csv(csv_path)
        print(f"Preprocessed data loaded from {csv_path}")
        return data
    except FileNotFoundError:
        print(f"No data found in: {csv_path}")
        return None


def main():
    # # Import and preprocess MNIST data
    # mnist_images, mnist_labels = import_MNIST_dataset()
    # mnist_data = load_preprocessed_data("csv/mnist_data.csv")
    # if mnist_data is None:
    #     mnist_data = preprocess_dataset(mnist_images, mnist_labels, "csv/mnist_data", purpose="training")

    # # Preprocess testing data (from the new digits folder)
    # test_data_path = "datasets/test_data/digits/*" 
    # test_data = load_preprocessed_data("csv/test_data.csv")
    # if test_data is None:
    #     test_data = preprocess_dataset(test_data_path, None, "csv/test_data", purpose="testing")

    # # early exit to make sure the test or train data is loaded correctly.
    # if mnist_data is None:
    #     print("Failed to load training or testing data. Exiting.")
    #     return
    
    # Train with digits
    training_data_path = "datasets/test_data/digits/*" 
    training_data = load_preprocessed_data("csv/digits_training_data.csv")
    if training_data is None:
        training_data = preprocess_dataset(training_data_path, None, "csv/digits_training_data", purpose="testing")

    # Prepare training data
    train_data = training_data.iloc[:, :-1]
    train_target = training_data.iloc[:, -1]

    # # Prepare training data
    # train_data = mnist_data.iloc[:, :-1]
    # train_target = mnist_data.iloc[:, -1]
    
    # Split into training and validation sets (using the training data)
    X_train, X_val, y_train, y_val = train_test_split(train_data, train_target, test_size=0.3, shuffle=False)
    

    # Check if test data exists
    # if test_data.empty:
    #     print("Failed to load test data. Exiting.")
    #     return
    
    # # Prepare testing data
    # test_data_features = test_data.iloc[:, :-1]
    # test_data_target = test_data.iloc[:, -1]
    
    # #set test data to be X_test and y_test
    # X_test, y_test = test_data_features, test_data_target


    # Check if models exist, if not, train them
    loaded_models = load_trained_models()
    if loaded_models is None:
        loaded_models = train_and_save_models(X_train, y_train)

    # Predict and evaluate
    predict_and_evaluate(X_val, y_val, loaded_models)
    # predict_and_evaluate(X_test, y_test, loaded_models)

    
def single():
    loaded_models = load_trained_models()

    # Predict single digit
    single_image_path = "datasets/test_data/digits/8/8_510.jpg"
    if os.path.exists(single_image_path):
        print(f"\nPredicting digit in: {single_image_path}")
        predict_single_digit(single_image_path, loaded_models)
    else:
        print(f"{single_image_path} does not exist")


if __name__ == "__main__":
    main()
    # single()
    
