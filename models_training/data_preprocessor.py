# ============================================
# FILE: models_training/data_preprocessor.py
# ============================================
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import pickle

# Configuration
IMG_SIZE = 224
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
DATA_PATH = '../data/raw/'
PROCESSED_PATH = '../data/processed/'


def load_data(data_split='Training'):
    """Loads images and labels from the specified split folder."""
    images = []
    labels = []
    split_path = os.path.join(DATA_PATH, data_split)

    print(f"Loading {data_split} data...")
    for label, class_name in enumerate(CLASSES):
        class_path = os.path.join(split_path, class_name)
        if not os.path.exists(class_path):
            print(f"Warning: Path {class_path} does not exist. Skipping.")
            continue

        for img_file in tqdm(os.listdir(class_path), desc=f'Loading {class_name}'):
            img_path = os.path.join(class_path, img_file)
            try:
                # Read image using OpenCV
                img = cv2.imread(img_path)
                if img is None:
                    continue
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Resize
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                # Normalize pixel values to [0, 1]
                img = img.astype(np.float32) / 255.0

                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue

    return np.array(images), np.array(labels)


def preprocess_and_save():
    """Main function to load, split, and save preprocessed data."""
    # Create processed directory if it doesn't exist
    os.makedirs(PROCESSED_PATH, exist_ok=True)

    # Load training data (we'll split this further into train/val)
    X_train_full, y_train_full = load_data('Training')

    # Split training data into train (80%) and validation (20%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=42,
        stratify=y_train_full
    )

    # Load test data
    X_test, y_test = load_data('Testing')

    # Convert labels to one-hot encoding
    num_classes = len(CLASSES)
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    # Save processed data
    print("Saving processed data...")
    with open(os.path.join(PROCESSED_PATH, 'train_data.pkl'), 'wb') as f:
        pickle.dump((X_train, y_train_cat, y_train), f)
    with open(os.path.join(PROCESSED_PATH, 'val_data.pkl'), 'wb') as f:
        pickle.dump((X_val, y_val_cat, y_val), f)
    with open(os.path.join(PROCESSED_PATH, 'test_data.pkl'), 'wb') as f:
        pickle.dump((X_test, y_test_cat, y_test), f)

    print(f"Data preprocessing complete!")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Image shape: {X_train[0].shape}")


if __name__ == "__main__":
    preprocess_and_save()

