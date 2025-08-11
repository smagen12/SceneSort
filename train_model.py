import tensorflow as tf
import os
import pickle 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import callbacks

#  CONFIG 
IMG_SIZE = 150  # smaller image size for custom CNN
BATCH_SIZE = 32
EPOCHS = 15
AUTO = tf.data.AUTOTUNE

CLASS_NAMES = tf.constant(['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street'])
NUM_CLASSES = len(CLASS_NAMES)

TRAIN_DIR = r"C:\Users\maayi\IFU\classification\Data\seg_train"
TEST_DIR = r"C:\Users\maayi\IFU\classification\Data\seg_test"

#LABEL PARSING
def get_label(path):
    path_str = tf.strings.regex_replace(path, "\\\\", "/")
    parts = tf.strings.split(path_str, '/')
    return tf.cast(parts[-2] == CLASS_NAMES, tf.int32)

# PREPROCESSING 
def load_and_preprocess(path, label, training=True):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0

    if training:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_contrast(img, 0.2, 1)

    return img, label

#  DATASET
def build_dataset(directory, training=True):
    pattern = os.path.join(directory, "*", "*.jpg")
    files = tf.data.Dataset.list_files(pattern, shuffle=training, seed=42)
    dataset = files.map(lambda x: (x, get_label(x)), num_parallel_calls=AUTO)
    dataset = dataset.map(lambda x, y: load_and_preprocess(x, y, training=training), num_parallel_calls=AUTO)
    if training:
        dataset = dataset.shuffle(1000)
    dataset = dataset.batch(BATCH_SIZE).prefetch(AUTO)
    return dataset

# CNN MODEL 
def build_custom_cnn():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# MAIN TRAINING LOOP
def main():
    print("[INFO] Preparing datasets...")
    train_ds = build_dataset(TRAIN_DIR, training=True)
    val_ds = build_dataset(TEST_DIR, training=False)

    print("[INFO] Building custom CNN...")
    model = build_custom_cnn()
    model.summary()

    # Callbacks
    checkpoint = callbacks.ModelCheckpoint("custom_model.h5", save_best_only=True, monitor='val_accuracy', verbose=1)
    early_stop = callbacks.EarlyStopping(patience=3, restore_best_weights=True)

    print("[INFO] Training custom CNN...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stop]
    )
    with open("training_history.pkl", "wb") as f:
        pickle.dump(history.history, f)
    print("[INFO] Training complete. Model saved as 'custom_model.h5'.")

if __name__ == "__main__":
    main()
