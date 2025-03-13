import numpy as np
import pandas as pd
import PIL.Image
from PIL import Image
import glob
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import seaborn as sns
import pickle
sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", 28)

# Preprocessing
def preprocess_training_data_to_df(path):
    data = []
    target = []
    i = 0
    for entry in glob.iglob(path):
        print(f"Processing class: {entry}")
        for image in glob.iglob(entry+'/*.jpg'):
            try:
                img = Image.open(image)
                img = img.resize((28, 28))
                img = img.convert('L')  # Convert to greyscale
                img = np.array(img)
                img = 255 - img  # Invert colors
                img = img / 255  # Scaling
                img = img.ravel()  # Flatten image
                img = img.tolist()
                data.append(img)
                target.append(i)
                print(f"Processed image: {image}")
            except Exception as e:
                print(f"Error processing image {image}: {e}")
        i += 1
    if not data:
        print("No images were processed.")
    data = pd.DataFrame(data)
    data["target"] = target
    data = data.sample(frac=1).reset_index(drop=True)  # Shuffling data

    
    data.to_csv("csv/digits_data.csv", index=False)  # index=False prevents saving row numbers
    print("Preprocessed data saved to digits_data.csv")

    return data

# Preprocessing
def preprocess_test_data_to_df(path):
    """
    Preprocesses image data from a single folder containing test images.
    Each image is resized, converted to greyscale, inverted, scaled, flattened, and
    saved to a CSV file named after the folder.

    Args:
        path (str): The path to the folder containing test images.
                     Example: "datasets/test_data/multi_digit_package"
    """
    folder_name = os.path.basename(path)  # Get the folder name for the CSV
    print(f"Processing test data from folder: {folder_name}")

    data = []
    target = []

    for image_path in glob.iglob(os.path.join(path, "*.png")):  # Iterate through .png files only
        try:
            img = Image.open(image_path)
            img = img.resize((8, 8))
            img = img.convert('L')  # Convert to greyscale
            img = np.array(img)
            img = 255 - img  # Invert colors
            img = img / 10  # Scaling
            img = img.ravel()  # Flatten image
            img = img.tolist()
            data.append(img)
            target.append(folder_name)  # Use folder name as target (for now, might need more refined labeling)
            print(f"Processed image: {image_path}")
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

    if not data:
        print(f"No images were processed in {folder_name}.")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["target"] = target
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffling data

    csv_filename = f"csv/{folder_name}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Preprocessed test data from {folder_name} saved to {csv_filename}")

    return df


def preprocess_single_image(image_path):
    """
    Preprocesses a single digit image.

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
        img = img / 255
        img = img.ravel()
        return img
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

def predict_single_digit(image_path, models):
    """
    Predicts the digit in a single image using the loaded models.

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
        print(f"{model_name.replace('_',' ').title()} prediction: {prediction[0]}")
    return predictions



# Models
def load_models():
    from sklearn import svm
    from sklearn.naive_bayes import GaussianNB
    from sklearn import tree
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import SGDClassifier
    from xgboost import XGBClassifier

    models = {
        # "svm": {
        #     "file": "models/svm_model.pkl",
        #     "class": svm.SVC(kernel='rbf', gamma=0.001, C=5),
        # },
        "gaussian_naive_bayes": {
            "file": "models/gaussian_naive_bayes_model.pkl",
            "class": GaussianNB(),
        },
        "decision_tree": {
            "file": "models/decision_tree_model.pkl",
            "class": tree.DecisionTreeClassifier(),
        },
        "random_forest": {
            "file": "models/random_forest_model.pkl",
            "class": RandomForestClassifier(max_depth=10, random_state=0),
        },
        "k_nearest_neighbors": {
            "file": "models/k_nearest_neighbors_model.pkl",
            "class": KNeighborsClassifier(n_neighbors=7, metric='euclidean'),
        },
        # "stochastic_gradient_decent": {
        #     "file": "models/stochastic_gradient_decent_model.pkl",
        #     "class": SGDClassifier(loss="hinge", penalty="l2", max_iter=100),
        # },
        "xgboost": {
            "file": "models/xgboost_model.pkl",
            "class": XGBClassifier(),
        },
    }

    # Load the trained models
    loaded_models = {}
    for model_name, model_data in models.items():
        try:
            with open(model_data["file"], "rb") as f:
                loaded_models[model_name] = pickle.load(f)
                print(f"Loaded {model_name} model")
        except FileNotFoundError:
            print(f"{model_name} model file not found.")
    
    return loaded_models

def load_models_and_predict(X_train, X_test, y_train, y_test):
    from sklearn import svm
    from sklearn.naive_bayes import GaussianNB
    from sklearn import tree
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import SGDClassifier
    from xgboost import XGBClassifier

    models = {
        # "svm": {
        #     "file": "models/svm_model.pkl",
        #     "class": svm.SVC(kernel='rbf', gamma=0.001, C=5),
        # },
        "gaussian_naive_bayes": {
            "file": "models/gaussian_naive_bayes_model.pkl",
            "class": GaussianNB(),
        },
        "decision_tree": {
            "file": "models/decision_tree_model.pkl",
            "class": tree.DecisionTreeClassifier(),
        },
        "random_forest": {
            "file": "models/random_forest_model.pkl",
            "class": RandomForestClassifier(max_depth=2, random_state=0),
        },
        "k_nearest_neighbors": {
            "file": "models/k_nearest_neighbors_model.pkl",
            "class": KNeighborsClassifier(n_neighbors=5, metric='euclidean'),
        },
        # "stochastic_gradient_decent": {
        #     "file": "models/stochastic_gradient_decent_model.pkl",
        #     "class": SGDClassifier(loss="hinge", penalty="l2", max_iter=5),
        # },
        "xgboost": {
            "file": "models/xgboost_model.pkl",
            "class": XGBClassifier(),
        },
    }

    # Load the trained models
    loaded_models = {}
    for model_name, model_data in models.items():
        try:
            with open(model_data["file"], "rb") as f:
                loaded_models[model_name] = pickle.load(f)
                print(f"Loaded {model_name} model")
        except FileNotFoundError:
            print(f"{model_name} model file not found. Training a new model.")
            model = model_data["class"]
            model.fit(X_train.values, y_train.values)
            loaded_models[model_name] = model  # Store the trained model
            with open(model_data["file"], "wb") as f:
                pickle.dump(model, f)
                print(f"Saved new {model_name} model")

    # Prediction and Accuracy
    for model_name, model in loaded_models.items():
        predicted = model.predict(X_test.values)
        accuracy = (len(X_test[predicted == y_test]) / len(X_test)) * 100
        print(f"{model_name.replace('_',' ').title()} Accuracy: {accuracy}")

    return loaded_models

# Load Training Data
def load_preprocessed_data():
    try:
        data = pd.read_csv("csv/digits_data.csv")
        print("Preprocessed data loaded from digits_data.csv")
        return data
    except FileNotFoundError:
        path = "datasets/training_data/digits/*"
        data = preprocess_training_data_to_df(path)
        return data

def main():
    pre = load_preprocessed_data()
    # test_data = preprocess_test_data_to_df("datasets/test_data/multi_digit_package")
    
    # pre = preprocess_training_data_to_df("digits/*")

    # For training
    data = pre.iloc[:,:-1]
    target = pre.iloc[:,-1]

    # For testing
    # data = test_data.iloc[:,:-1]
    # target = test_data.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, shuffle=False)


    loaded_models = load_models_and_predict(X_train, X_test, y_train, y_test)
    # loaded_models = load_models()

    # Predict single digit
    single_image_path = "digits/7/7_18140.jpg"  # Replace with your image path
    if os.path.exists(single_image_path):
      print(f"\nPredicting digit in: {single_image_path}")
      predict_single_digit(single_image_path, loaded_models)
    else:
      print(f"{single_image_path} does not exist")

main()