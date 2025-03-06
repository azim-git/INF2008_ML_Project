import cv2
import numpy as np
from PIL import Image
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt

# --- Preprocessing ---
def preprocess_single_digit(image: Image.Image) -> np.ndarray:
    """Preprocesses a single digit image for the model."""
    img = image.resize((28, 28)).convert('L')
    img_array = np.array(img)
    img_array = 255 - img_array 
    img_array = img_array / 10
    return img_array.ravel()

def extract_digits_from_image(image_path, debug=False):
    """
    Extracts individual digit images and decimal points from a multi-digit image using OpenCV.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return [], [], []

    # --- 1. Preprocessing: Adaptive Thresholding and Dilation ---
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    if debug:
        # --- Visualization 1: Thresholded and Dilated Image ---
        cv2.imshow("Thresholded and Dilated", dilated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # --- End Visualization 1 ---

    # --- 2. Contour Detection ---
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if debug:
        # --- Visualization 2: Contours on Original Image ---
        img_with_contours = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 2)
        cv2.imshow("Contours on Original Image", img_with_contours)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # --- End Visualization 2 ---

    # --- 3. Contour Filtering: Digits and Decimal Points ---
    digit_images = []
    digit_boxes = []
    decimal_point_boxes = []
    all_boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        all_boxes.append((x,y,w,h))

    if debug:
        # --- Visualization 3: Bounding Boxes ---
        img_with_boxes = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        for x, y, w, h in all_boxes:
            cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 255), 2) # yellow for all boxes
        cv2.imshow("Bounding Boxes", img_with_boxes)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # --- End Visualization 3 ---

    # --- 4. Determine Decimal Points Based on Size Ratio ---
    decimal_point_boxes, digit_boxes = refine_decimal_point_detection(all_boxes, digit_boxes, decimal_point_boxes)

    if debug:
        # --- Visualization 3.1: Bounding Boxes - Redefined ---
        img_with_boxes = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for x, y, w, h in digit_boxes:
            cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 0, 255), 2)
        for x, y, w, h in decimal_point_boxes:
            cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green for decimal point
        cv2.imshow("Bounding Boxes - Redefined", img_with_boxes)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # --- End Visualization 3.1 ---

    # --- 5. Merge and Adjust Overlapping Boxes ---
    digit_boxes = merge_and_adjust_overlapping_boxes(digit_boxes)

    if debug:
        # --- Visualization 4: Merged Bounding Boxes ---
        img_with_merged_boxes = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for x, y, w, h in digit_boxes:
            cv2.rectangle(img_with_merged_boxes, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow("Merged and Adjusted Bounding Boxes", img_with_merged_boxes)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # --- End Visualization 4 ---

    # --- 6. Sorting (Left-to-Right) ---
    digit_boxes.sort()
    decimal_point_boxes.sort()

    # --- 7. Digit Extraction with Padding and Squaring---
    if debug:
        img_with_padding = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for x, y, w, h in digit_boxes:
        digit_roi_squared = extract_digit_with_padding_and_squaring(img, x, y, w, h)
        if digit_roi_squared is not None:
            digit_pil = Image.fromarray(digit_roi_squared)
            digit_images.append(digit_pil)
            if debug:
                # Draw original box in red and padded box in blue
                padding_percent = 0.2  # Increased to 20% for more padding
                padding_x = int(w * padding_percent)
                padding_y = int(h * padding_percent)

                # Adjust Bounding Box
                x_start = max(0, x - padding_x)
                y_start = max(0, y - padding_y)
                x_end = min(img.shape[1], x + w + padding_x)
                y_end = min(img.shape[0], y + h + padding_y)
                cv2.rectangle(img_with_padding, (x, y), (x + w, y + h), (0, 0, 255), 1)
                cv2.rectangle(img_with_padding, (x_start, y_start), (x_end, y_end), (255, 0, 0), 1)

    if debug:
        cv2.imshow("Padding on Extracted Digits", img_with_padding)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return digit_images, digit_boxes, decimal_point_boxes

def refine_decimal_point_detection(all_boxes, digit_boxes, decimal_point_boxes):
    """
    Refines decimal point detection by considering both size and aspect ratio.
    """
    digit_areas = [w * h for x, y, w, h in all_boxes]
    if not digit_areas:
        return [], all_boxes

    avg_area = sum(digit_areas) / len(digit_areas)

    # Classify based on size and aspect ratio
    new_decimal_point_boxes = []
    new_digit_boxes = []

    for x, y, w, h in all_boxes:
        box_area = w * h
        aspect_ratio = h / w if w > 0 else float("inf")  # Prevent division by zero

        # A decimal point should be small and nearly square
        if box_area < avg_area / 2 and 0.5 < aspect_ratio < 2:  
            new_decimal_point_boxes.append((x, y, w, h))
        else:
            new_digit_boxes.append((x, y, w, h))

    return new_decimal_point_boxes, new_digit_boxes

def extract_digit_with_padding_and_squaring(img, x, y, w, h):
    # ... (rest of the function remains unchanged)
    padding_percent = 0.2
    padding_x = int(w * padding_percent)
    padding_y = int(h * padding_percent)

    x_start = max(0, x - padding_x)
    y_start = max(0, y - padding_y)
    x_end = min(img.shape[1], x + w + padding_x)
    y_end = min(img.shape[0], y + h + padding_y)

    digit_roi = img[y_start:y_end, x_start:x_end]

    # Create a square image
    side = max(digit_roi.shape)
    square_digit = np.full((side, side), 255, dtype=np.uint8)  # White background

    # Center the digit in the square
    x_offset = (side - digit_roi.shape[1]) // 2
    y_offset = (side - digit_roi.shape[0]) // 2
    
    #check if the offset is negative
    if x_offset < 0 or y_offset < 0:
        return None

    square_digit[y_offset:y_offset + digit_roi.shape[0], x_offset:x_offset + digit_roi.shape[1]] = digit_roi

    return square_digit

def classify_digits(digit_images, model):
    # ... (rest of the function remains unchanged)
    predictions = []
    data = []
    for image in digit_images:
        data.append(preprocess_single_digit(image))

    data = pd.DataFrame(data)

    if not data.empty:
        predictions = model.predict(data.values)

    return predictions

def combine_predictions(predictions, digit_boxes, decimal_point_boxes):
    """Combines digit predictions and inserts a decimal point if found."""
    if not isinstance(predictions, np.ndarray):
        return ""
    if predictions.size == 0:
        return ""

    combined_number = ""
    digit_boxes_with_predictions = list(zip(predictions, digit_boxes))
    digit_boxes_with_predictions.sort(key=lambda x: x[1][0])  # Sort by x-coordinate of digit boxes

    decimal_inserted = False
    if decimal_point_boxes:
        decimal_x, decimal_y, decimal_w, decimal_h  = decimal_point_boxes[0]

    for i, (prediction, (x, y, w, h)) in enumerate(digit_boxes_with_predictions):
        combined_number += str(prediction)
        if decimal_point_boxes and not decimal_inserted:
            # Check if the decimal point's x-coordinate is between this digit and the next
            for j in range(i+1,len(digit_boxes_with_predictions)):
                next_prediction, (next_x, next_y, next_w, next_h) = digit_boxes_with_predictions[j]
                if x < decimal_x < next_x and y + h / 2 < decimal_y:
                    combined_number += "."
                    decimal_inserted = True
                    break
    return combined_number

def load_trained_model(model_path):
    # ... (rest of the function remains unchanged)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def merge_and_adjust_overlapping_boxes(boxes):
    # ... (rest of the function remains unchanged)
    if len(boxes) <= 1:
        return boxes

    boxes.sort()  # Sort by x-coordinate

    merged_boxes = []
    used = [False] * len(boxes)

    for i in range(len(boxes)):
        if used[i]:
            continue

        x1, y1, w1, h1 = boxes[i]
        x2_max, y2_max, w2_max, h2_max = x1, y1, w1, h1

        # Check for overlapping boxes and adjust their positions
        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue

            x2, y2, w2, h2 = boxes[j]
            overlap_x = max(x1, x2)
            overlap_x_end = min(x1+w1, x2+w2)
            overlap_y = max(y1, y2)
            overlap_y_end = min(y1 + h1, y2 + h2)
            
            if overlap_x < overlap_x_end and overlap_y < overlap_y_end:  # Check for overlap
                used[j] = True
                # Adjust the right box
                boxes[j] = (x2_max+w2_max,y2, w2,h2)
            
            # Merging
            if overlap_x < overlap_x_end and max(y1,y2) < min(y1 + h1, y2 + h2):
                x2_max = min(x1,x2)
                y2_max = min(y1,y2)
                w2_max = max(x1+w1, x2+w2) - x2_max
                h2_max = max(y1+h1, y2+h2) - y2_max
                
                used[j] = True
                boxes[i] = (x2_max, y2_max, w2_max, h2_max)
                

        if not used[i]:
            merged_boxes.append((boxes[i][0], boxes[i][1],boxes[i][2], boxes[i][3]))

    return merged_boxes

def main():
    model_path = "models/k_nearest_neighbors_model.pkl"  # Change model if needed
    image_paths = [
        "datasets/test_data/hand-drawn/4-36.png",  
        "datasets/test_data/hand-drawn/4-36-5.png",
        "datasets/test_data/hand-drawn/-89.png",
        "datasets/test_data/hand-drawn/2431.png",
        "datasets/test_data/hand-drawn/2341.png",
        "datasets/test_data/hand-drawn/55900.png",
    ] 
    model = load_trained_model(model_path)

    for image_path in image_paths:
        print(f"\nProcessing: {image_path}")
        digit_images, digit_boxes, decimal_point_boxes = extract_digits_from_image(image_path, debug=True)

        print(f"{len(digit_images)} digits detected in {os.path.basename(image_path)}")
        print(f"{len(decimal_point_boxes)} potential decimal points detected in {os.path.basename(image_path)}")

        if not digit_images:
            print("No digits were detected in the image.")
            continue

        predictions = classify_digits(digit_images, model)

        if not isinstance(predictions, np.ndarray):
          print("No predictions were made on the image.")
          continue

        if  predictions.size == 0:
            print("No predictions were made on the image.")
            continue

        plt.figure(figsize=(len(digit_images) * 2, 5))
        for i, (digit_image, prediction) in enumerate(zip(digit_images, predictions)):
            plt.subplot(1, len(digit_images), i + 1)
            plt.imshow(digit_image, cmap='gray')
            plt.title(f"Prediction: {prediction}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()

        multi_digit_number = combine_predictions(predictions, digit_boxes, decimal_point_boxes)
        print(f"The multi-digit number in the image is: {multi_digit_number}")

if __name__ == "__main__":
    main()
