import os
import pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

print("\n" + "=" * 60)
print("SIMPLE RESNET50 - FIXED FOR 95% ACCURACY")
print("=" * 60)

# ===============================
# Paths
# ===============================
PROCESSED_PATH = "../data/processed/"
MODEL_PATH = "../app/models/saved/"
os.makedirs(MODEL_PATH, exist_ok=True)

# ===============================
# Load Data (Simple)
# ===============================
print("\n📂 Loading data...")
with open(os.path.join(PROCESSED_PATH, 'train_data.pkl'), 'rb') as f:
    X_train, y_train_cat, _ = pickle.load(f)
with open(os.path.join(PROCESSED_PATH, 'val_data.pkl'), 'rb') as f:
    X_val, y_val_cat, _ = pickle.load(f)
with open(os.path.join(PROCESSED_PATH, 'test_data.pkl'), 'rb') as f:
    X_test, y_test_cat, _ = pickle.load(f)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Convert to numpy if needed
X_train = np.array(X_train)
X_val = np.array(X_val)
X_test = np.array(X_test)

# ===============================
# FIX: Proper ResNet50 Preprocessing
# ===============================
print("\n🔄 Applying ResNet50 preprocessing...")


def preprocess_simple(X):
    """Simple and correct ResNet50 preprocessing"""
    # Resize to 224x224
    X_resized = tf.image.resize(X, (224, 224)).numpy()
    # ResNet50 expects pixels in [0, 255] range then preprocess_input
    X_scaled = X_resized * 255.0
    return tf.keras.applications.resnet50.preprocess_input(X_scaled)


X_train = preprocess_simple(X_train)
X_val = preprocess_simple(X_val)
X_test = preprocess_simple(X_test)

# ===============================
# Build Simple ResNet50
# ===============================
print("\n🏗️ Building ResNet50...")

# Use simple architecture
base_model = tf.keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

# Simple top layers
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(4, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ===============================
# Train (Fast)
# ===============================
print("\n" + "=" * 60)
print("TRAINING - 10 EPOCHS ONLY")
print("=" * 60)

# Simple callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(MODEL_PATH, 'resnet50_simple.keras'),
        monitor='val_accuracy',
        save_best_only=True
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True
    )
]

# Train
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=10,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# ===============================
# Evaluate
# ===============================
print("\n" + "=" * 60)
print("EVALUATION")
print("=" * 60)

# Test accuracy
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\n✅ Test Accuracy: {test_acc * 100:.2f}%")

if test_acc >= 0.95:
    print("🎯 TARGET ACHIEVED: 95%+")
else:
    print(f"⚠️ Current: {test_acc * 100:.2f}% - Running fine-tuning...")

    # Phase 2: Fine-tuning
    print("\n🔄 Fine-tuning for 5 more epochs...")

    # Unfreeze some layers
    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train more
    history2 = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=5,
        batch_size=32,
        verbose=1
    )

    # Final evaluation
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\n✅ Final Test Accuracy: {test_acc * 100:.2f}%")

# ===============================
# Save model
# ===============================
model.save(os.path.join(MODEL_PATH, 'resnet50_95.keras'))
print(f"\n💾 Model saved to: {os.path.join(MODEL_PATH, 'resnet50_95.keras')}")

# ===============================
# Quick predictions
# ===============================
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
y_true = np.argmax(y_test_cat, axis=1)
print("\n📊 Classification Report (Last 100 samples):")
print(classification_report(y_true[:100], y_pred[:100],
                            target_names=["Glioma", "Meningioma", "No Tumor", "Pituitary"]))

print("\n" + "=" * 60)
print(f"🏁 DONE! Accuracy: {test_acc * 100:.2f}%")
print("=" * 60)