# augment_forged.py
import os
import cv2
import numpy as np

# ---------- PATHS ----------
FORGED_PATH = "preprocessed/forged"       # folder with existing preprocessed forged images
AUG_PATH = "preprocessed/forged_aug"     # folder to save augmented images
os.makedirs(AUG_PATH, exist_ok=True)

# ---------- AUGMENTATION FUNCTION ----------
def augment_image(img):
    augmented = []
    
    # Original
    augmented.append(img)
    
    # Rotation ±5° and ±10°
    for angle in [-10, -5, 5, 10]:
        M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1)
        rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        augmented.append(rotated)
    
    # Slight shift
    for dx, dy in [(-5,0), (5,0), (0,-5), (0,5)]:
        M = np.float32([[1,0,dx],[0,1,dy]])
        shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        augmented.append(shifted)
    
    # Add Gaussian noise
    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    noisy = cv2.add(img, noise)
    augmented.append(noisy)
    
    return augmented

# ---------- PROCESS ----------
for file in os.listdir(FORGED_PATH):
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(FORGED_PATH, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        aug_images = augment_image(img)
        
        base_name = os.path.splitext(file)[0]
        for idx, aug in enumerate(aug_images):
            save_path = os.path.join(AUG_PATH, f"{base_name}_aug{idx}.jpg")
            cv2.imwrite(save_path, aug)

print(f"✅ Augmentation completed! Augmented images saved in {AUG_PATH}")
