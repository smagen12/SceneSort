import os
import pickle 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model

# CONFIG 
IMG_SIZE = 150
BATCH_SIZE = 32
CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
TEST_DIR = r"C:\Users\maayi\IFU\classification\Data\seg_test"
MODEL_PATH = "custom_model.h5"

# CATEGORY DISTRIBUTION PIE CHART 
category_counts = {}
sample_images = {}

for category in CLASS_NAMES:
    folder_path = os.path.join(TEST_DIR, category)
    if os.path.exists(folder_path):
        images = os.listdir(folder_path)
        category_counts[category] = len(images)
    else:
        print(f" Folder not found: {folder_path}")
        category_counts[category] = 0

# Plot pie chart
labels = list(category_counts.keys())
sizes = list(category_counts.values())

plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title("Category Distribution in Test Set (Pie Chart)")
plt.axis('equal')  # Ensures pie is circular
plt.tight_layout()
plt.show()


#PREPARE DATASET
def get_label(path):
    path_str = tf.strings.regex_replace(path, "\\\\", "/")
    parts = tf.strings.split(path_str, '/')
    return tf.cast(parts[-2] == CLASS_NAMES, tf.int32)

def load_and_preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

pattern = os.path.join(TEST_DIR, "*", "*.jpg")
files = tf.data.Dataset.list_files(pattern, shuffle=False)
labeled_files = files.map(lambda x: (x, get_label(x)))
test_ds = labeled_files.map(load_and_preprocess).batch(BATCH_SIZE)

#LOAD MODEL AND PREDICT
model = load_model(MODEL_PATH)

y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(np.argmax(labels.numpy(), axis=1))


# PLOT ACCURACY AND LOSS
with open(r"C:\Users\maayi\IFU\classification\model\training_history.pkl", "rb") as f:
    history = pickle.load(f) 

plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# CONFUSION MATRIX 
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# CLASSIFICATION REPORT
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
