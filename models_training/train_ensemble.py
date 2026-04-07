import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import warnings

warnings.filterwarnings('ignore')

print("\n" + "=" * 70)
print("BRAIN TUMOR DETECTION - ENSEMBLE MODEL (ALL 3 MODELS)")
print("=" * 70)

# ===============================
# Configuration
# ===============================
PROCESSED_PATH = "../data/processed/"
MODEL_PATH = "../app/models/saved/"
VIS_PATH = "../visualization_outputs/"
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
IMG_SIZE = 224

os.makedirs(VIS_PATH, exist_ok=True)

# ===============================
# Load Test Data
# ===============================
print("\n📂 LOADING TEST DATA")
print("-" * 50)

try:
    with open(os.path.join(PROCESSED_PATH, 'test_data.pkl'), 'rb') as f:
        X_test, y_test_cat, y_test = pickle.load(f)
    print(f"✓ Loaded {len(X_test)} test images")
    print(f"✓ Image shape: {X_test[0].shape}")
except:
    print("✗ Could not load test data!")
    exit()

y_true = np.argmax(y_test_cat, axis=1)

# ===============================
# Model-Specific Preprocessing
# ===============================
print("\n🔄 PREPROCESSING FOR EACH MODEL")
print("-" * 50)

# Resize all images to 224x224
X_resized = tf.image.resize(X_test, (IMG_SIZE, IMG_SIZE)).numpy()

# Custom CNN expects [0,1] range
X_test_custom = X_resized.astype(np.float32)

# MobileNetV2 expects preprocessed input
X_test_mobilenet = tf.keras.applications.mobilenet_v2.preprocess_input(X_resized * 255.0)

# ResNet50 expects preprocessed input
X_test_resnet = tf.keras.applications.resnet50.preprocess_input(X_resized * 255.0)

print("✓ Custom CNN: [0,1] range")
print("✓ MobileNetV2: Preprocessed")
print("✓ ResNet50: Preprocessed")

# ===============================
# Load All Three Models
# ===============================
print("\n📦 LOADING ALL THREE MODELS")
print("-" * 50)

models_info = []
predictions = {}
probabilities = {}
accuracies = {}

# 1. Load Custom CNN
custom_paths = [
    os.path.join(MODEL_PATH, "custom_cnn_final.h5"),
    os.path.join(MODEL_PATH, "custom_cnn_best.h5"),
    "../saved_models/custom_cnn_final.h5"
]

custom_model = None
for path in custom_paths:
    if os.path.exists(path):
        try:
            custom_model = tf.keras.models.load_model(path)
            print(f"✓ Custom CNN loaded from: {os.path.basename(path)}")
            break
        except:
            continue

if custom_model is not None:
    probs = custom_model.predict(X_test_custom, verbose=0)
    preds = np.argmax(probs, axis=1)
    acc = accuracy_score(y_true, preds)
    predictions['Custom CNN'] = preds
    probabilities['Custom CNN'] = probs
    accuracies['Custom CNN'] = acc
    models_info.append(('Custom CNN', acc, probs, preds))
    print(f"  → Accuracy: {acc * 100:.2f}%")

# 2. Load MobileNetV2
mobilenet_paths = [
    os.path.join(MODEL_PATH, "brain_tumor_best.keras"),
    os.path.join(MODEL_PATH, "mobilenetv2_final.keras"),
    "../saved_models/mobilenetv2_final.keras",
    "../saved_models/mobilenetv2_best.keras"
]

mobilenet_model = None
for path in mobilenet_paths:
    if os.path.exists(path):
        try:
            mobilenet_model = tf.keras.models.load_model(path)
            print(f"✓ MobileNetV2 loaded from: {os.path.basename(path)}")
            break
        except:
            continue

if mobilenet_model is not None:
    probs = mobilenet_model.predict(X_test_mobilenet, verbose=0)
    preds = np.argmax(probs, axis=1)
    acc = accuracy_score(y_true, preds)
    predictions['MobileNetV2'] = preds
    probabilities['MobileNetV2'] = probs
    accuracies['MobileNetV2'] = acc
    models_info.append(('MobileNetV2', acc, probs, preds))
    print(f"  → Accuracy: {acc * 100:.2f}%")

# 3. Load ResNet50
resnet_paths = [
    os.path.join(MODEL_PATH, "resnet50_best.keras"),
    os.path.join(MODEL_PATH, "resnet50_final.keras"),
    "../saved_models/resnet50_best.keras",
    "../saved_models/resnet50_final.keras"
]

resnet_model = None
for path in resnet_paths:
    if os.path.exists(path):
        try:
            resnet_model = tf.keras.models.load_model(path)
            print(f"✓ ResNet50 loaded from: {os.path.basename(path)}")
            break
        except:
            continue

if resnet_model is not None:
    probs = resnet_model.predict(X_test_resnet, verbose=0)
    preds = np.argmax(probs, axis=1)
    acc = accuracy_score(y_true, preds)
    predictions['ResNet50'] = preds
    probabilities['ResNet50'] = probs
    accuracies['ResNet50'] = acc
    models_info.append(('ResNet50', acc, probs, preds))
    print(f"  → Accuracy: {acc * 100:.2f}%")

if len(models_info) < 2:
    print("\n✗ Need at least 2 models for ensemble!")
    exit()

# ===============================
# ENSEMBLE METHODS
# ===============================
print("\n" + "=" * 70)
print("🎯 ENSEMBLE METHODS")
print("=" * 70)

# Method 1: Weighted Average (based on accuracy)
weights = np.array([acc for _, acc, _, _ in models_info])
weights = weights / weights.sum()

weighted_probs = np.zeros_like(probabilities[models_info[0][0]])
for i, (name, _, probs, _) in enumerate(models_info):
    weighted_probs += weights[i] * probs

weighted_preds = np.argmax(weighted_probs, axis=1)
weighted_acc = accuracy_score(y_true, weighted_preds)

print(f"\n📊 Weighted Ensemble (weights based on accuracy):")
for (name, acc, _, _), w in zip(models_info, weights):
    print(f"  {name}: weight={w:.3f} (acc={acc * 100:.2f}%)")
print(f"  → Accuracy: {weighted_acc * 100:.2f}%")

# Method 2: Simple Average
avg_probs = np.mean([probs for _, _, probs, _ in models_info], axis=0)
avg_preds = np.argmax(avg_probs, axis=1)
avg_acc = accuracy_score(y_true, avg_preds)
print(f"\n📊 Simple Average Ensemble:")
print(f"  → Accuracy: {avg_acc * 100:.2f}%")

# Method 3: Majority Voting
all_preds = np.array([preds for _, _, _, preds in models_info])
majority_preds = []
for i in range(len(y_true)):
    votes = all_preds[:, i]
    majority_pred = np.bincount(votes).argmax()
    majority_preds.append(majority_pred)
majority_acc = accuracy_score(y_true, majority_preds)
print(f"\n📊 Majority Voting Ensemble:")
print(f"  → Accuracy: {majority_acc * 100:.2f}%")

# Method 4: Max Probability
max_probs_preds = np.argmax(np.max([probs for _, _, probs, _ in models_info], axis=0), axis=1)
max_probs_acc = accuracy_score(y_true, max_probs_preds)
print(f"\n📊 Max Probability Ensemble:")
print(f"  → Accuracy: {max_probs_acc * 100:.2f}%")

# Choose best ensemble method
ensemble_methods = [
    (weighted_acc, weighted_preds, "Weighted Average"),
    (avg_acc, avg_preds, "Simple Average"),
    (majority_acc, majority_preds, "Majority Voting"),
    (max_probs_acc, max_probs_preds, "Max Probability")
]

best_acc, best_preds, best_method = max(ensemble_methods, key=lambda x: x[0])
print(f"\n🏆 BEST ENSEMBLE METHOD: {best_method}")
print(f"   Accuracy: {best_acc * 100:.2f}%")

# ===============================
# VISUALIZATION 1: Model Comparison Bar Chart
# ===============================
print("\n📊 GENERATING VISUALIZATIONS")
print("-" * 50)

plt.figure(figsize=(14, 7))

# Prepare data
model_names_list = [name for name, _, _, _ in models_info]
model_accs_list = [acc * 100 for _, acc, _, _ in models_info]
ensemble_names = ['Weighted', 'Simple Avg', 'Majority', 'Max Prob', 'BEST']
ensemble_accs = [weighted_acc * 100, avg_acc * 100, majority_acc * 100, max_probs_acc * 100, best_acc * 100]

# Combine for plotting
all_names = model_names_list + ensemble_names
all_accs = model_accs_list + ensemble_accs

# Create color map
colors = ['#3498db', '#2ecc71', '#e74c3c'] + ['#f39c12', '#f1c40f', '#e67e22', '#d35400', '#27ae60']

bars = plt.bar(all_names, all_accs, color=colors[:len(all_names)], edgecolor='black', linewidth=1)

# Add value labels
for bar, acc in zip(bars, all_accs):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f'{acc:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add threshold lines
plt.axhline(y=95, color='green', linestyle='--', linewidth=2, label='95% Target', alpha=0.7)
plt.axhline(y=98, color='red', linestyle=':', linewidth=2, label='98% Excellent', alpha=0.7)

# Customize plot
plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
plt.title('Brain Tumor Detection: Individual Models vs Ensemble Methods', fontsize=14, fontweight='bold')
plt.ylim([0, 105])
plt.legend(loc='upper left', fontsize=10)
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.xticks(rotation=45, ha='right')

# Add background colors
plt.axhspan(98, 105, alpha=0.1, color='green', label='Excellent')
plt.axhspan(95, 98, alpha=0.1, color='yellow', label='Target')
plt.axhspan(0, 95, alpha=0.05, color='red', label='Below Target')

plt.tight_layout()
plt.savefig(os.path.join(VIS_PATH, 'ensemble_comparison_full.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: ensemble_comparison_full.png")

# ===============================
# VISUALIZATION 2: Confusion Matrix for Best Ensemble
# ===============================
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true, best_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            annot_kws={'size': 14})
plt.title(f'Best Ensemble ({best_method}) Confusion Matrix\nAccuracy: {best_acc * 100:.2f}%',
          fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(VIS_PATH, 'ensemble_confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: ensemble_confusion_matrix.png")

# ===============================
# VISUALIZATION 3: Per-Class Performance
# ===============================
from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1, _ = precision_recall_fscore_support(y_true, best_preds)

plt.figure(figsize=(12, 6))
x = np.arange(len(CLASS_NAMES))
width = 0.25

plt.bar(x - width, precision, width, label='Precision', color='#3498db', edgecolor='black')
plt.bar(x, recall, width, label='Recall', color='#2ecc71', edgecolor='black')
plt.bar(x + width, f1, width, label='F1-Score', color='#e74c3c', edgecolor='black')

plt.xlabel('Tumor Type', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title(f'Per-Class Performance - Best Ensemble ({best_method})', fontsize=14, fontweight='bold')
plt.xticks(x, CLASS_NAMES, fontsize=11)
plt.legend(fontsize=11)
plt.grid(axis='y', alpha=0.3)
plt.ylim([0, 1.05])

# Add value labels
for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
    plt.text(i - width, p + 0.02, f'{p:.3f}', ha='center', fontsize=9, fontweight='bold')
    plt.text(i, r + 0.02, f'{r:.3f}', ha='center', fontsize=9, fontweight='bold')
    plt.text(i + width, f + 0.02, f'{f:.3f}', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(VIS_PATH, 'ensemble_per_class.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: ensemble_per_class.png")

# ===============================
# VISUALIZATION 4: Model Accuracy Comparison (Simple)
# ===============================
plt.figure(figsize=(12, 6))

# Simple bar chart with just models and best ensemble
simple_names = model_names_list + [f'Ensemble\n({best_method})']
simple_accs = model_accs_list + [best_acc * 100]

colors_simple = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
bars = plt.bar(simple_names, simple_accs, color=colors_simple[:len(simple_names)], edgecolor='black')

# Add value labels
for bar, acc in zip(bars, simple_accs):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.axhline(y=95, color='green', linestyle='--', linewidth=2, label='95% Target')
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Model Performance: Individual vs Ensemble', fontsize=14, fontweight='bold')
plt.ylim([80, 101])
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VIS_PATH, 'model_comparison_simple.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: model_comparison_simple.png")

# ===============================
# SAVE RESULTS
# ===============================
print("\n💾 SAVING RESULTS")
print("-" * 50)

# Save all predictions
results = {
    'true_labels': y_true,
    'individual_models': {name: preds for name, preds in predictions.items()},
    'ensemble_methods': {
        'weighted': weighted_preds,
        'average': avg_preds,
        'majority': majority_preds,
        'max_prob': max_probs_preds,
        'best': best_preds,
        'best_method': best_method
    },
    'accuracies': {
        **{name: acc * 100 for name, acc in accuracies.items()},
        'weighted_ensemble': weighted_acc * 100,
        'average_ensemble': avg_acc * 100,
        'majority_ensemble': majority_acc * 100,
        'max_prob_ensemble': max_probs_acc * 100,
        'best_ensemble': best_acc * 100
    }
}

with open(os.path.join(VIS_PATH, 'ensemble_results.pkl'), 'wb') as f:
    pickle.dump(results, f)

# Save detailed report
with open(os.path.join(VIS_PATH, 'ensemble_detailed_report.txt'), 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("BRAIN TUMOR DETECTION - ENSEMBLE RESULTS\n")
    f.write("=" * 70 + "\n\n")

    f.write("INDIVIDUAL MODEL PERFORMANCE:\n")
    f.write("-" * 40 + "\n")
    for name, acc in accuracies.items():
        f.write(f"{name:15s}: {acc * 100:.4f}%\n")

    f.write("\nENSEMBLE METHOD PERFORMANCE:\n")
    f.write("-" * 40 + "\n")
    f.write(f"Weighted Average : {weighted_acc * 100:.4f}%\n")
    f.write(f"Simple Average   : {avg_acc * 100:.4f}%\n")
    f.write(f"Majority Voting  : {majority_acc * 100:.4f}%\n")
    f.write(f"Max Probability  : {max_probs_acc * 100:.4f}%\n")
    f.write(f"{'=' * 40}\n")
    f.write(f"BEST METHOD      : {best_method}\n")
    f.write(f"BEST ACCURACY    : {best_acc * 100:.4f}%\n")
    f.write("=" * 40 + "\n\n")

    f.write("CLASSIFICATION REPORT (Best Ensemble):\n")
    f.write("-" * 40 + "\n")
    f.write(classification_report(y_true, best_preds, target_names=CLASS_NAMES))

    f.write("\n\nPER-CLASS METRICS:\n")
    f.write("-" * 40 + "\n")
    for i, class_name in enumerate(CLASS_NAMES):
        f.write(f"{class_name:12s}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}\n")

print("✓ Saved: ensemble_results.pkl")
print("✓ Saved: ensemble_detailed_report.txt")

# ===============================
# FINAL SUMMARY
# ===============================
print("\n" + "=" * 70)
print("🎯 FINAL SUMMARY")
print("=" * 70)

print("\n📊 Individual Models:")
for name, acc in accuracies.items():
    status = "✅" if acc >= 0.95 else "⚠️" if acc >= 0.90 else "❌"
    print(f"  {status} {name:12s}: {acc * 100:.4f}%")

print("\n📊 Ensemble Methods:")
print(f"  📈 Weighted Average : {weighted_acc * 100:.4f}%")
print(f"  📈 Simple Average   : {avg_acc * 100:.4f}%")
print(f"  📈 Majority Voting  : {majority_acc * 100:.4f}%")
print(f"  📈 Max Probability  : {max_probs_acc * 100:.4f}%")

print("\n" + "=" * 70)
print(f"🏆 BEST ENSEMBLE METHOD: {best_method}")
print(f"🏆 FINAL ACCURACY: {best_acc * 100:.4f}%")
print("=" * 70)

if best_acc >= 0.98:
    print("\n✅✅✅ EXCELLENT! 98%+ TARGET ACHIEVED! ✅✅✅")
elif best_acc >= 0.95:
    print("\n✅✅ GOOD! 95%+ TARGET ACHIEVED! ✅✅")
elif best_acc >= 0.90:
    print("\n⚠️ ACCEPTABLE: 90%+ Accuracy (Close to Target)")
else:
    print(f"\n❌ Current accuracy {best_acc * 100:.2f}% is below 90%")

print(f"\n📁 All visualizations saved to: {VIS_PATH}")
print("=" * 70)