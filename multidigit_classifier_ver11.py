import cv2
import numpy as np
from PIL import Image
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def preprocess_single_digit(image: Image.Image) -> np.ndarray:
    img = image.resize((28, 28)).convert('L')
    img_array = np.array(img)
    img_array = 255 - img_array
    img_array = img_array / 255.0
    return img_array.ravel()

def extract_digits_from_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return [], [], []

    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    processed_img = cv2.dilate(thresh, kernel, iterations=2)  # Improved dilation only

    contours, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_boxes = [cv2.boundingRect(cnt) for cnt in contours]
    all_boxes = split_wide_boxes(img, all_boxes)

    decimal_point_boxes, digit_boxes = refine_decimal_point_detection(all_boxes)
    digit_boxes = merge_and_adjust_overlapping_boxes(digit_boxes)
    digit_boxes.sort()
    decimal_point_boxes.sort()

    decimal_point_boxes = filter_stray_decimal_points(decimal_point_boxes, digit_boxes)

    digit_images = [Image.fromarray(extract_digit_with_padding_and_squaring(img, x, y, w, h))
                    for x, y, w, h in digit_boxes]
    return digit_images, digit_boxes, decimal_point_boxes, img

def split_wide_boxes(img, boxes):
    new_boxes = []
    for x, y, w, h in boxes:
        aspect_ratio = w / h
        if aspect_ratio > 1.5:  # Likely two digits
            vertical_projection = np.sum(img[y:y+h, x:x+w], axis=0)
            peaks, _ = find_peaks(-vertical_projection, distance=w//4)
            if len(peaks) > 0:
                split_index = peaks[0]
                new_boxes.append((x, y, split_index, h))
                new_boxes.append((x + split_index, y, w - split_index, h))
            else:
                new_boxes.append((x, y, w, h))
        else:
            new_boxes.append((x, y, w, h))
    return new_boxes


def refine_decimal_point_detection(all_boxes):
    digit_areas = [w * h for _, _, w, h in all_boxes]
    if not digit_areas:
        return [], all_boxes
    avg_area = sum(digit_areas) / len(digit_areas)
    decimal_point_boxes, digit_boxes = [], []
    for x, y, w, h in all_boxes:
        aspect_ratio = h / w if w > 0 else float("inf")
        if w * h < avg_area / 2 and 0.5 < aspect_ratio < 2:
            decimal_point_boxes.append((x, y, w, h))
        else:
            digit_boxes.append((x, y, w, h))
    return decimal_point_boxes, digit_boxes

def filter_stray_decimal_points(decimal_point_boxes, digit_boxes):
    """Removes decimal points that are positioned above the top half of any digit."""
    valid_decimal_points = []
    for dx, dy, dw, dh in decimal_point_boxes:
        is_valid = False
        for x, y, w, h in digit_boxes:
            top_half_threshold = y + (h // 2)
            if dy > top_half_threshold:
                is_valid = True
                break
        if is_valid:
            valid_decimal_points.append((dx, dy, dw, dh))
    return valid_decimal_points

def extract_digit_with_padding_and_squaring(img, x, y, w, h):
    padding = int(max(w, h) * 0.2)
    x_start, y_start = max(0, x - padding), max(0, y - padding)
    x_end, y_end = min(img.shape[1], x + w + padding), min(img.shape[0], y + h + padding)
    digit_roi = img[y_start:y_end, x_start:x_end]
    side = max(digit_roi.shape)
    square_digit = np.full((side, side), 255, dtype=np.uint8)
    x_offset, y_offset = (side - digit_roi.shape[1]) // 2, (side - digit_roi.shape[0]) // 2
    square_digit[y_offset:y_offset + digit_roi.shape[0], x_offset:x_offset + digit_roi.shape[1]] = digit_roi
    return square_digit

def classify_digits(digit_images, model):
    data = [preprocess_single_digit(image) for image in digit_images]
    return model.predict(pd.DataFrame(data).values) if data else []

def combine_predictions(predictions, digit_boxes, decimal_point_boxes):
    if not isinstance(predictions, np.ndarray) or predictions.size == 0:
        return ""
    combined_number, decimal_inserted = "", False
    digit_boxes_with_predictions = sorted(zip(predictions, digit_boxes), key=lambda x: x[1][0])
    decimal_x = decimal_point_boxes[0][0] if decimal_point_boxes else None
    for i, (prediction, (x, _, _, _)) in enumerate(digit_boxes_with_predictions):
        combined_number += str(prediction)
        if decimal_x and not decimal_inserted and i < len(digit_boxes_with_predictions) - 1:
            if x < decimal_x < digit_boxes_with_predictions[i + 1][1][0]:
                combined_number += "."
                decimal_inserted = True
    return combined_number

def merge_and_adjust_overlapping_boxes(boxes):
    if len(boxes) <= 1:
        return boxes

    boxes = sorted(boxes, key=lambda b: b[0])
    merged_boxes = []

    for i in range(len(boxes)):
        if i in merged_boxes:
            continue

        x1, y1, w1, h1 = boxes[i]
        for j in range(i + 1, len(boxes)):
            x2, y2, w2, h2 = boxes[j]

            if x2 < x1 + w1 and x2 + w2 > x1:  # Overlapping horizontally
                new_x = min(x1, x2)
                new_y = min(y1, y2)
                new_w = max(x1 + w1, x2 + w2) - new_x
                new_h = max(y1 + h1, y2 + h2) - new_y
                boxes[i] = (new_x, new_y, new_w, new_h)
                merged_boxes.append(j)

        merged_boxes.append(i)
    return [boxes[i] for i in range(len(boxes)) if i in merged_boxes]

def visualize_results(image, digit_boxes, decimal_point_boxes, predictions):
    plt.figure(figsize=(10, 5))
    img_with_boxes = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h), prediction in zip(digit_boxes, predictions):
        cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_with_boxes, str(prediction), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    for x, y, w, h in decimal_point_boxes:
        cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 0, 255), 2)
    plt.imshow(img_with_boxes, cmap='gray')
    plt.axis('off')
    plt.show()

def main():
    model_path = "models/k_nearest_neighbors_model.pkl"
    image_paths = [
        "curated_dataset/hand-drawn/4-36.png",  
        # "curated_dataset/hand-drawn/4-36-5.png",
        # "curated_dataset/hand-drawn/-89.png",
        # "curated_dataset/hand-drawn/666.png",
        # "curated_dataset/hand-drawn/2341.png",
        "curated_dataset/hand-drawn/55900.png",
        "curated_dataset/multi_digit_package/0000123.png",
        # "curated_dataset/multi_digit_package/9184875.png",
    ] 
    model = pickle.load(open(model_path, "rb"))
    for image_path in image_paths:
        digit_images, digit_boxes, decimal_point_boxes, img = extract_digits_from_image(image_path)
        predictions = classify_digits(digit_images, model)
        result = combine_predictions(predictions, digit_boxes, decimal_point_boxes)
        print(f"Predicted Number in {os.path.basename(image_path) }: {result}")
        visualize_results(img, digit_boxes, decimal_point_boxes, predictions)

if __name__ == "__main__":
    main()
