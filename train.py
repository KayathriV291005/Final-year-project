# train.py
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# ---------- PATHS ----------
PREPROCESSED_PATH = "preprocessed"
MODEL_PATH = "model/signature_model_v2.h5"
IMAGE_SIZE = (100, 200)
CHANNELS = 1

# ---------- LOAD DATA ----------
def load_data():
    images = []
    labels = []
    
    for label, folder in enumerate(["forged", "genuine"]):  # 0=forged, 1=genuine
        folder_path = os.path.join(PREPROCESSED_PATH, folder)
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = cv2.imread(os.path.join(folder_path, file), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, IMAGE_SIZE)
                img = img / 255.0
                images.append(img)
                labels.append(label)
    
    X = np.array(images).reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1], CHANNELS)
    y = np.array(labels)
    return X, y

# ---------- LOAD AND SPLIT ----------
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ---------- CLASS WEIGHTS ----------
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))
print("✅ Class Weights:", class_weights)

# ---------- MODEL ----------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], CHANNELS)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ---------- TRAIN ----------
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=20,
                    batch_size=16,
                    class_weight=class_weights)

# ---------- SAVE ----------
model.save(MODEL_PATH)
print(f"✅ Model saved successfully at: {MODEL_PATH}")
