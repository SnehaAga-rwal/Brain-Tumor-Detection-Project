# ============================================
# FILE: models_training/train_custom_cnn.py
# ============================================
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import warnings
warnings.filterwarnings('ignore')

# ===============================
# Configuration
# ===============================
PROCESSED_PATH = '../data/processed/'
IMG_SIZE = 224    # Standard for MobileNetV2
NUM_CLASSES = 4
BATCH_SIZE = 16   # Optimized for memory and convergence
EPOCHS = 30       # Fewer epochs but high quality training
LEARNING_RATE = 0.001 
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Create directories
os.makedirs('../app/models/saved', exist_ok=True)
os.makedirs('../visualization_outputs', exist_ok=True)

# ===============================
# Load data
# ===============================
print("="*60)
print("LOADING DATA")
print("="*60)

with open(os.path.join(PROCESSED_PATH, 'train_data.pkl'), 'rb') as f:
    X_train, y_train_cat, _ = pickle.load(f)

with open(os.path.join(PROCESSED_PATH, 'val_data.pkl'), 'rb') as f:
    X_val, y_val_cat, _ = pickle.load(f)

with open(os.path.join(PROCESSED_PATH, 'test_data.pkl'), 'rb') as f:
    X_test, y_test_cat, y_test = pickle.load(f)

# Resize and Preprocess
def prepare_data(X, size):
    X_resized = tf.image.resize(X, (size, size)).numpy()
    # MobileNetV2 expects [-1, 1] range
    return preprocess_input(X_resized * 255.0)

print(f"Preparing data (Resizing to {IMG_SIZE} and Preprocessing)...")
X_train = prepare_data(X_train, IMG_SIZE)
X_val = prepare_data(X_val, IMG_SIZE)
X_test = prepare_data(X_test, IMG_SIZE)

# ===============================
# Build Optimized Custom CNN (MobileNetV2 based)
# ===============================
def create_optimized_cnn():
    # Use Functional API for better control
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_tensor=inputs
    )
    
    # Unfreeze the top layers for better fine-tuning
    base_model.trainable = True
    for layer in base_model.layers[:-40]: 
        layer.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="Optimized_Custom_CNN_98Plus")
    
    return model

# ===============================
# Training Setup
# ===============================
model = create_optimized_cnn()

# Learning rate scheduler: Cosine Decay
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=LEARNING_RATE,
    decay_steps=EPOCHS * (len(X_train) // BATCH_SIZE),
    alpha=0.0001
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

# tf.data pipeline for maximum performance
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train_cat)).shuffle(2000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val_cat)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

callbacks_list = [
    callbacks.ModelCheckpoint('../app/models/saved/custom_cnn_best.h5', monitor='val_accuracy', save_best_only=True, verbose=1, mode='max'),
    callbacks.EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1, mode='max')
]

# ===============================
# Train
# ===============================
print("\nSTARTING OPTIMIZED TRAINING...")
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks_list,
    verbose=1
)

# ===============================
# Final Evaluation
# ===============================
print("\nFINAL EVALUATION...")
test_results = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"Test Accuracy: {test_results[1]*100:.2f}%")

# Predictions
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
y_true = np.argmax(y_test_cat, axis=1)

# Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title(f'Custom CNN Confusion Matrix (Acc: {test_results[1]:.4f})')
plt.savefig('../visualization_outputs/custom_cnn_confusion_matrix.png')

# Classification Report
report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
plt.figure(figsize=(10, 6))
sns.heatmap(np.array([[report[cls][m] for m in ['precision', 'recall', 'f1-score']] for cls in CLASS_NAMES]),
            annot=True, fmt='.2f', cmap='YlGnBu', xticklabels=['Precision', 'Recall', 'F1-Score'], yticklabels=CLASS_NAMES)
plt.title('Custom CNN Classification Report')
plt.savefig('../visualization_outputs/custom_cnn_classification_report.png')

model.save('../app/models/saved/custom_cnn_final.h5')
print("\n✅ DONE!")
