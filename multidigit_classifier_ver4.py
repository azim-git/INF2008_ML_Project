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

def extract_digits_from_image(image_path, debug=False):
    """
    Extracts individual digit images from a multi-digit image using OpenCV.

    Args:
        image_path (str): Path to the multi-digit image.
        debug (bool): If True, displays intermediate images for debugging.

    Returns:
        list: A list of PIL Image objects, each representing an extracted digit.
        list: A list of bounding box coordinates (x, y, w, h) for each digit.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return [], []

    # 1. Preprocessing: Improved Thresholding and Dilation (Conditional)
    
    blurred = cv2.GaussianBlur(img, (5, 5), 0) # Apply Gaussian blur to smooth image.
    
    ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) # Using Otsu's method

    kernel = np.ones((3, 3), np.uint8)
    # if the image has lots of gaps in the digits, add the dilation:
    # dilated = cv2.dilate(thresh, kernel, iterations=1) # dilate image to fill in gaps
    dilated = thresh

    if debug:
        # --- Visualization 1: Thresholded and Dilated Image ---
        cv2.imshow("Thresholded and Dilated", dilated)
        cv2.waitKey(0)  # Wait for a key press
        cv2.destroyAllWindows()
        # --- End Visualization 1 ---

    # 2. Contour Detection: Improved
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if debug:
        # --- Visualization 2: Contours on Original Image ---
        img_with_contours = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convert to BGR for color drawing
        cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 2)  # Draw contours in green
        cv2.imshow("Contours on Original Image", img_with_contours)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # --- End Visualization 2 ---

    # 3. Contour Filtering and Bounding Boxes: Refined
    digit_images = []
    digit_boxes = []
    
    max_boxes = 10
    box_count = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Refined filtering:
        if w > 2 and h > 5 and w < 70 and h < 70 and h / w > 0.5:
            # Check if the bounding box is not too close to the image edges
            # if x > 2 and y > 2 and x + w < img.shape[1] - 2 and y + h < img.shape[0] - 2:
            digit_boxes.append((x, y, w, h))
            box_count += 1
            if box_count >= max_boxes:
                break

    if debug:
        # --- Visualization 3: Bounding Boxes ---
        img_with_boxes = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for x, y, w, h in digit_boxes:
            cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw rectangle in red
        cv2.imshow("Bounding Boxes", img_with_boxes)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # --- End Visualization 3 ---

    # 4. Merge Overlapping Boxes:
    # if there are too many boxes, try merging them
    if len(digit_boxes) > 7:
        digit_boxes = merge_overlapping_boxes(digit_boxes)

    if debug:
        # --- Visualization 4: Merged Bounding Boxes ---
        img_with_merged_boxes = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for x, y, w, h in digit_boxes:
            cv2.rectangle(img_with_merged_boxes, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw rectangle in blue
        cv2.imshow("Merged Bounding Boxes", img_with_merged_boxes)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # --- End Visualization 4 ---

    # 5. Sorting (Left-to-Right)
    digit_boxes.sort()  # Sort by x-coordinate (leftmost first)

    # 6. Digit Extraction
    # --- Visualization 5: Padding on Extracted Digits ---
    if debug:
        img_with_padding = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for x, y, w, h in digit_boxes:
        # --- New: Calculate Padding ---
        padding_percent = 0.1  # Adjust this value (e.g., 0.1 for 10%, 0.2 for 20%)
        padding_x = int(w * padding_percent)
        padding_y = int(h * padding_percent)

        # --- New: Adjust Bounding Box ---
        x_start = max(0, x - padding_x)
        y_start = max(0, y - padding_y)
        x_end = min(img.shape[1], x + w + padding_x)
        y_end = min(img.shape[0], y + h + padding_y)

        # Draw original box in red
        if debug:
            cv2.rectangle(img_with_padding, (x, y), (x + w, y + h), (0, 0, 255), 1) # original box in red
            cv2.rectangle(img_with_padding, (x_start, y_start), (x_end, y_end), (255, 0, 0), 1)  # Padded box in blue

        # --- New: Extract Digit with Padding ---
        digit_roi = img[y_start:y_end, x_start:x_end]
        
        digit_pil = Image.fromarray(digit_roi)
        digit_images.append(digit_pil)
    if debug:
        cv2.imshow("Padding on Extracted Digits", img_with_padding)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # --- End Visualization 5 ---
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

def merge_overlapping_boxes(boxes):
    """
    Merges overlapping bounding boxes.

    Args:
        boxes (list): A list of bounding boxes (x, y, w, h).

    Returns:
        list: A list of merged bounding boxes.
    """
    
    if len(boxes) <= 1:
        return boxes

    merged_boxes = []
    used = [False] * len(boxes)

    for i in range(len(boxes)):
        if used[i]:
            continue

        x1, y1, w1, h1 = boxes[i]
        x2_max, y2_max, w2_max, h2_max = x1, y1, w1, h1

        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue

            x2, y2, w2, h2 = boxes[j]

            # Check for overlap (simplified for demonstration)
            if max(x1, x2) < min(x1 + w1, x2 + w2) and max(y1, y2) < min(y1 + h1, y2 + h2):
                x2_max = min(x1,x2)
                y2_max = min(y1,y2)
                w2_max = max(x1+w1, x2+w2) - x2_max
                h2_max = max(y1+h1, y2+h2) - y2_max

                used[j] = True
        
        merged_boxes.append((x2_max, y2_max, w2_max, h2_max))

    return merged_boxes

def main():
    model_path = "models/k_nearest_neighbors_model.pkl"
    image_path = "datasets/test_data/multi_digit_package/0000943.png"  # Replace with your multi-digit image path
    
    #Load trained model
    model = load_trained_model(model_path)

    # Extract single digit images and their bounding boxes
    digit_images, digit_boxes = extract_digits_from_image(image_path, debug=True)
    
    print(f"{len(digit_images)} digits detected in {os.path.basename(image_path)}")

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
