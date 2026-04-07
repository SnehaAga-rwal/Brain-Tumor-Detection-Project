import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

print("\n" + "=" * 60)
print("BRAIN TUMOR MobileNetV2 TRAINING (OPTIMIZED FOR 95%+ ACCURACY)")
print("=" * 60)

# ==============================
# SETTINGS - OPTIMIZED
# ==============================
DATASET_PATH = "../data/raw"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
EPOCHS_HEAD = 20  # Increased
EPOCHS_FINE = 20  # Increased
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_FINE = 1e-5

print(f"\n📁 Dataset path: {os.path.abspath(DATASET_PATH)}")
print(f"🎯 Target: >95% accuracy")

# ==============================
# LOAD DATASET WITH VALIDATION SPLIT
# ==============================
print("\n📊 Loading datasets...")

# Load training data with validation split
full_train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_PATH, "Training"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    label_mode='int'
)

# Split training into train (80%) and validation (20%)
train_size = int(0.8 * len(full_train_ds))
val_size = len(full_train_ds) - train_size

train_ds = full_train_ds.take(train_size)
val_ds = full_train_ds.skip(train_size)

# Load test data
test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_PATH, "Testing"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode='int'
)

class_names = full_train_ds.class_names
NUM_CLASSES = len(class_names)
print(f"\n✅ Classes: {class_names}")
print(f"✅ Training batches: {train_size}")
print(f"✅ Validation batches: {val_size}")
print(f"✅ Test batches: {len(test_ds)}")

# ==============================
# OPTIMIZE DATA PIPELINE
# ==============================
AUTOTUNE = tf.data.AUTOTUNE


def optimize_dataset(ds):
    return ds.cache().prefetch(AUTOTUNE)


train_ds = optimize_dataset(train_ds)
val_ds = optimize_dataset(val_ds)
test_ds = optimize_dataset(test_ds)

# ==============================
# ENHANCED DATA AUGMENTATION
# ==============================
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomBrightness(0.1),
])

# ==============================
# BUILD OPTIMIZED MODEL
# ==============================
print("\n🏗️ Building optimized MobileNetV2...")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))

# Augmentation
x = data_augmentation(inputs)

# Preprocessing
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

# Base model
x = base_model(x, training=False)

# Global pooling
x = tf.keras.layers.GlobalAveragePooling2D()(x)

# Batch normalization
x = tf.keras.layers.BatchNormalization()(x)

# Dense layers with regularization
x = tf.keras.layers.Dense(1024, activation='relu',
                          kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Dense(512, activation='relu',
                          kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.BatchNormalization()(x)

# Output layer
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

# ==============================
# COMPILE WITH OPTIMIZER
# ==============================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_HEAD),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==============================
# ADVANCED CALLBACKS
# ==============================
callbacks = [
    # Model checkpoint
    tf.keras.callbacks.ModelCheckpoint(
        '../saved_models/mobilenetv2_best.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),

    # Early stopping
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        mode='max',
        verbose=1
    ),

    # Reduce learning rate
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),

    # Learning rate scheduler
    tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: LEARNING_RATE_HEAD * (0.95 ** epoch)
    )
]

# ==============================
# PHASE 1: TRAIN HEAD
# ==============================
print("\n" + "=" * 60)
print("PHASE 1: TRAINING CLASSIFIER HEAD")
print("=" * 60)

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
    callbacks=callbacks,
    verbose=1
)

# ==============================
# PHASE 2: FINE-TUNING
# ==============================
print("\n" + "=" * 60)
print("PHASE 2: FINE-TUNING")
print("=" * 60)

# Unfreeze base model
base_model.trainable = True

# Freeze early layers (keep more layers trainable for brain tumor features)
for layer in base_model.layers[:-50]:  # Unfreeze more layers
    layer.trainable = False

print(f"\nTrainable layers: {sum(1 for layer in base_model.layers if layer.trainable)}/{len(base_model.layers)}")

# Recompile with lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_FINE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Update callbacks for fine-tuning
fine_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        '../saved_models/mobilenetv2_finetuned.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        restore_best_weights=True,
        mode='max',
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINE,
    callbacks=fine_callbacks,
    verbose=1
)

# ==============================
# EVALUATION
# ==============================
print("\n" + "=" * 60)
print("📊 FINAL EVALUATION")
print("=" * 60)

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
print(f"\n✅ Test Accuracy: {test_accuracy * 100:.4f}%")

# Generate predictions for detailed report
print("\n📋 Generating Classification Report...")
y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    preds = np.argmax(preds, axis=1)
    y_pred.extend(preds)
    y_true.extend(labels.numpy())

# Print classification report
print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print(report)

# ==============================
# PLOT TRAINING HISTORY
# ==============================
print("\n📈 Generating training plots...")

# Combine histories
all_loss = history1.history['loss'] + history2.history['loss']
all_val_loss = history1.history['val_loss'] + history2.history['val_loss']
all_acc = history1.history['accuracy'] + history2.history['accuracy']
all_val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']

# Plot accuracy
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(all_acc, label='Training Accuracy', linewidth=2)
plt.plot(all_val_acc, label='Validation Accuracy', linewidth=2)
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Target')
plt.title('Model Accuracy', fontsize=14)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(all_loss, label='Training Loss', linewidth=2)
plt.plot(all_val_loss, label='Validation Loss', linewidth=2)
plt.title('Model Loss', fontsize=14)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../visualization_outputs/mobilenetv2_training_history.png', dpi=100)
plt.close()

# ==============================
# SAVE FINAL MODEL
# ==============================
os.makedirs("../saved_models", exist_ok=True)
os.makedirs("../app/models/saved", exist_ok=True)

# Save in multiple formats
model.save("../saved_models/mobilenetv2_final.keras")
model.save("../app/models/saved/mobilenetv2_final.keras")
model.save("../app/models/saved/brain_tumor_best.keras")  # For ensemble compatibility

# Save results
with open('../visualization_outputs/mobilenetv2_results.txt', 'w') as f:
    f.write("MOBILENETV2 TRAINING RESULTS\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Test Accuracy: {test_accuracy * 100:.4f}%\n\n")
    f.write("Classification Report:\n")
    f.write(report)

# ==============================
# FINAL SUMMARY
# ==============================
print("\n" + "=" * 60)
print("🎯 FINAL SUMMARY")
print("=" * 60)
print(f"\n✅ Model: MobileNetV2")
print(f"✅ Test Accuracy: {test_accuracy * 100:.4f}%")

if test_accuracy >= 0.98:
    print("🏆 EXCELLENT! Model achieved 98%+ accuracy!")
elif test_accuracy >= 0.95:
    print("✅ GOOD! Model achieved 95%+ target accuracy!")
elif test_accuracy >= 0.93:
    print("⚠️ ACCEPTABLE: Model achieved 93%+ accuracy (close to target)")
else:
    print(f"❌ Model accuracy {test_accuracy * 100:.2f}% is below target")

print(f"\n💾 Models saved:")
print(f"   - ../saved_models/mobilenetv2_final.keras")
print(f"   - ../app/models/saved/brain_tumor_best.keras")
print(f"\n📊 Visualizations saved to: ../visualization_outputs/")
print("\n" + "=" * 60)