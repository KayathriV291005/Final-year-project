# augment_genuine.py
import os
import cv2
import numpy as np

GENUINE_PATH = "preprocessed/genuine"
AUG_PATH = "preprocessed/genuine_aug"
os.makedirs(AUG_PATH, exist_ok=True)

def augment_image(img):
    augmented = []
    augmented.append(img)  # original

    # Small rotations
    for angle in [-5, 5]:
        M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1)
        rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        augmented.append(rotated)

    # Small shifts
    for dx, dy in [(-2,0), (2,0), (0,-2), (0,2)]:
        M = np.float32([[1,0,dx],[0,1,dy]])
        shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        augmented.append(shifted)

    return augmented

# Process all genuine images
for file in os.listdir(GENUINE_PATH):
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(GENUINE_PATH, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        aug_images = augment_image(img)
        base_name = os.path.splitext(file)[0]
        for idx, aug in enumerate(aug_images):
            save_path = os.path.join(AUG_PATH, f"{base_name}_aug{idx}.jpg")
            cv2.imwrite(save_path, aug)

print(f"✅ Genuine augmentation completed! Saved in {AUG_PATH}")
