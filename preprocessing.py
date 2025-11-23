
import cv2
import numpy as np
import os

def preprocess_image(image_path, output_size=(200, 100)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error loading image: {image_path}")
        return None
    img = cv2.resize(img, output_size)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return img

if __name__ == "__main__":
    input_folders = ["dataset/genuine", "dataset/forged"]
    output_folders = ["preprocessed/genuine", "preprocessed/forged"]

    for in_folder, out_folder in zip(input_folders, output_folders):
        os.makedirs(out_folder, exist_ok=True)
        for filename in os.listdir(in_folder):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                img_path = os.path.join(in_folder, filename)
                processed = preprocess_image(img_path)
                if processed is not None:
                    cv2.imwrite(os.path.join(out_folder, filename), processed)

    print("✅ Preprocessing completed successfully!")
