# test.py
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ---------- PATH SETTINGS ----------
MODEL_PATH = "model/signature_model_v2.h5"
TEST_FOLDER = "test_signatures"   # Folder where test images are kept
IMAGE_SIZE = (100, 200)           # same as training
CHANNELS = 1                      # grayscale

# ---------- LOAD TRAINED MODEL ----------
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# ---------- PREPROCESS FUNCTION ----------
def preprocess_image(image_path):
    """Preprocess test signature image same as training"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.resize(img, IMAGE_SIZE)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img_norm = img / 255.0
    return img_norm.reshape(1, IMAGE_SIZE[0], IMAGE_SIZE[1], CHANNELS)

# ---------- TESTING FUNCTION ----------
def test_signatures():
    threshold = 0.7   # decision boundary (you can adjust 0.6–0.8)
    print("\n🔍 Starting Signature Verification...\n")

    # Loop through all test images
    for filename in os.listdir(TEST_FOLDER):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(TEST_FOLDER, filename)
            processed = preprocess_image(img_path)
            
            # Predict probabilities
            prediction = model.predict(processed)[0]
            prob_forged, prob_genuine = prediction[0], prediction[1]

            # Apply threshold
            if prob_genuine >= threshold:
                result = f"Genuine ✅ (p={prob_genuine:.3f})"
            else:
                result = f"Forged ❌ (p={prob_genuine:.3f})"

            print(f"{filename} → {result}")

    print("\n✅ Testing completed!")

# ---------- MAIN ----------
if __name__ == "__main__":
    test_signatures()
