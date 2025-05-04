import cv2
import numpy as np
import joblib
import os

# Load models
model_path = os.path.join(os.path.dirname(__file__), 'svm_crack.pkl')
svm_crack = joblib.load(model_path)
model_path = os.path.join(os.path.dirname(__file__), 'svm_pothole.pkl')
svm_pothole = joblib.load(model_path)
lane_mask_dir = "C:/Users/subha/Downloads/Cracks-and-Potholes-in-Road-Images/v1/Lanes"

def calculate_mask_coverage(mask_path, image_shape):
    if mask_path and os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (image_shape[1], image_shape[0]))
        return np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
    return 0.0

def extract_features(image, image_filename):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [128], [0, 256]).flatten()
    hist_norm = hist / np.sum(hist)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges) / (gray.shape[0] * gray.shape[1])
    crack_cov = pothole_cov = 0.0
    lane_filename = image_filename.replace(".jpg", "_LANE.jpg")
    lane_path = os.path.join(lane_mask_dir, lane_filename)
    lane_cov = calculate_mask_coverage(lane_path, image.shape)
    return np.hstack([hist_norm, edge_density, crack_cov, pothole_cov, lane_cov])

def localize_and_highlight(image, crack_present, pothole_present):
    overlay = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if crack_present:
        edges = cv2.Canny(gray, 100, 200)
        overlay[edges > 0] = [255, 255, 0]
    if pothole_present:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:
                cv2.drawContours(overlay, [cnt], -1, (255, 0, 0), 2)
    return overlay

def predict_image(image_path):
    image = cv2.imread(image_path)
    filename = os.path.basename(image_path)
    features = extract_features(image, filename).reshape(1, -1)
    
    crack_pred = svm_crack.predict(features)[0]
    pothole_pred = svm_pothole.predict(features)[0]
    
    result_img = localize_and_highlight(image, crack_pred, pothole_pred)
    
    output_filename = f"result_{filename}"
    output_path = os.path.join("outputs", output_filename)
    cv2.imwrite(output_path, result_img)

    label = f"Crack: {'Yes' if crack_pred else 'No'}, Pothole: {'Yes' if pothole_pred else 'No'}"
    return output_path, label

