import cv2
import numpy as np
from PIL import Image
import pickle
import os
import pandas as pd

def preprocess_single_digit(image):
    """
    Preprocesses an extracted single digit image.
    Each image is resized, converted to greyscale, inverted, scaled, and flattened.

    Args:
        image: A single digit image in PIL format.
    """
    img = image
    img = img.resize((8, 8))
    img = img.convert('L')  # Convert to greyscale
    img = np.array(img)
    img = 255 - img  # Invert colors
    img = img / 10  # Scaling
    img = img.ravel()  # Flatten image
    return img

def extract_digits_from_image(image_path):
    """
    Extracts individual digit images from a multi-digit image using OpenCV.

    Args:
        image_path (str): Path to the multi-digit image.

    Returns:
        list: A list of PIL Image objects, each representing an extracted digit.
        list: A list of bounding box coordinates (x, y, w, h) for each digit.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return [], []

    # 1. Preprocessing
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # 2. Contour Detection
    contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 3. Contour Filtering and Bounding Boxes
    digit_images = []
    digit_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 5 and h > 10:  # Basic size filter to exclude noise
            digit_boxes.append((x, y, w, h))

    # 4. Sorting (Left-to-Right)
    digit_boxes.sort()  # Sort by x-coordinate (leftmost first)

    # 5. Digit Extraction
    for x, y, w, h in digit_boxes:
        digit_roi = img[y : y + h, x : x + w]
        digit_pil = Image.fromarray(digit_roi)
        digit_images.append(digit_pil)

    return digit_images, digit_boxes

def classify_digits(digit_images, model):
    """
    Classifies a list of digit images using a trained model.

    Args:
        digit_images (list): A list of digit images (PIL Image objects).
        model: The trained model.

    Returns:
        list: A list of predicted digit labels.
    """
    predictions = []
    data = []
    for image in digit_images:
        data.append(preprocess_single_digit(image))
    
    data = pd.DataFrame(data)
    
    if not data.empty:
        predictions = model.predict(data.values)
        
    return predictions

def combine_predictions(predictions):
    """
    Combines a list of predicted digits into a multi-digit number.

    Args:
        predictions (list): A list of predicted digit labels.

    Returns:
        str: The combined multi-digit number.
    """
    return "".join(map(str, predictions))

def load_trained_model(model_path):
    """
    Loads a trained model from a .pkl file.
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def main():
    model_path = "models/k_nearest_neighbors_model.pkl"
    image_path = "datasets/test_data/multi_digit_package/9110231.png"  # Replace with your multi-digit image path
    
    #Load trained model
    model = load_trained_model(model_path)

    # Extract single digit images and their bounding boxes
    digit_images, digit_boxes = extract_digits_from_image(image_path)

    # Check if any images were extracted
    if not digit_images:
        print("No digits were detected in the image.")
        return

    # Classify the digits
    predictions = classify_digits(digit_images, model)

    # Check if any predictions were made
    if not predictions.size:
        print("No predictions were made on the image.")
        return

    # Combine the predictions
    multi_digit_number = combine_predictions(predictions)
    print(f"The multi-digit number in the image is: {multi_digit_number}")

if __name__ == "__main__":
    main()
