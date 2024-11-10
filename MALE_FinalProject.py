
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import tf_keras

# Load the Cassava dataset
(ds_train, ds_validation, ds_test), ds_info = tfds.load(
    name='cassava',
    split=['train', 'validation', 'test'],
    with_info=True,
    as_supervised=True
)

print(f"Number of training examples: {ds_info.splits['train'].num_examples}")
print(f"Number of validation examples: {ds_info.splits['validation'].num_examples}")
print(f"Number of test examples: {ds_info.splits['test'].num_examples}")
print(f"Number of classes: {ds_info.features['label'].num_classes}")
print(f"Class names: {ds_info.features['label'].names}")

# Image size for MobileNetV3
IMG_SIZE = 224

def preprocess_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image, label

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32

def prepare_dataset(ds, train=False):
    ds = ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    if train:
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    ds = ds.shuffle(1000) if train else ds
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)
    return ds

train_ds = prepare_dataset(ds_train, train=True)
val_ds = prepare_dataset(ds_validation)
test_ds = prepare_dataset(ds_test)

# Create the model
def create_model():
    model_url = "https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5"
    base_model = hub.KerasLayer(model_url, trainable=False)

    num_classes = ds_info.features['label'].num_classes

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    return model

# Learning rate schedule
def lr_schedule(epoch):
    lr = 0.001
    if epoch > 5:
        lr *= 0.1
    return lr

# Implement k-fold cross-validation
def k_fold_cross_validation(k=5):
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    
    all_scores = []
    for fold, (train_indices, val_indices) in enumerate(kfold.split(ds_train)):
        print(f"Fold {fold + 1}/{k}")
        
        model = create_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        
        # Create datasets for this fold
        train_data = tf.data.Dataset.from_tensor_slices(train_indices).map(lambda x: next(iter(ds_train.skip(x))))
        val_data = tf.data.Dataset.from_tensor_slices(val_indices).map(lambda x: next(iter(ds_train.skip(x))))
        
        train_data = prepare_dataset(train_data, train=True)
        val_data = prepare_dataset(val_data)
        
        # Train the model
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=10,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
                tf.keras.callbacks.LearningRateScheduler(lr_schedule)
            ]
        )
        
        # Evaluate the model
        scores = model.evaluate(val_data)
        all_scores.append(scores[1])  # Append accuracy
        
    print(f"Average accuracy across all folds: {np.mean(all_scores):.2f}")

# Run k-fold cross-validation
k_fold_cross_validation()

# Train the final model
model = create_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    ]
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test accuracy: {test_accuracy:.2f}")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Model Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Model Loss')
plt.show()

# Save the model
model.save('plant_disease_model.h5')
print("Model saved as 'plant_disease_model.h5'")

# Function to predict on a single image
def predict_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0

    predictions = model.predict(img_array)
    predicted_class = tf.argmax(predictions[0]).numpy()
    class_name = ds_info.features['label'].names[predicted_class]
    confidence = tf.reduce_max(predictions[0]).numpy()

    return class_name, confidence

# Visualize sample images and their predictions
def visualize_predictions(num_images=5):
    plt.figure(figsize=(15, 3*num_images))
    for i, (image, label) in enumerate(test_ds.take(num_images)):
        ax = plt.subplot(num_images, 3, i*3 + 1)
        plt.imshow(image[0])
        plt.title(f"Actual: {ds_info.features['label'].names[label[0]]}")
        plt.axis('off')
        
        predictions = model.predict(image)
        predicted_class = tf.argmax(predictions[0]).numpy()
        confidence = tf.reduce_max(predictions[0]).numpy()
        
        ax = plt.subplot(num_images, 3, i*3 + 2)
        plt.bar(range(len(ds_info.features['label'].names)), predictions[0])
        plt.title(f"Predicted: {ds_info.features['label'].names[predicted_class]}")
        plt.xticks(range(len(ds_info.features['label'].names)), ds_info.features['label'].names, rotation=90)
        
        ax = plt.subplot(num_images, 3, i*3 + 3)
        plt.text(0.5, 0.5, f"Confidence: {confidence:.2f}", ha='center', va='center', fontsize=12)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

visualize_predictions()

# Plot confusion matrix
def plot_confusion_matrix():
    y_pred = []
    y_true = []

    for image_batch, label_batch in test_ds:
        predictions = model.predict(image_batch)
        y_pred.extend(tf.argmax(predictions, axis=1).numpy())
        y_true.extend(label_batch.numpy())

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    print(classification_report(y_true, y_pred, target_names=ds_info.features['label'].names))

plot_confusion_matrix()

# GUI implementation
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        image = image.resize((300, 300))
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo
        
        class_name, confidence = predict_image(file_path)
        result_label.config(text=f"Predicted class: {class_name}\nConfidence: {confidence:.2f}")

# Create the main window
root = tk.Tk()
root.title("Plant Disease Classifier")

# Create and pack widgets
open_button = tk.Button(root, text="Open Image", command=open_image)
open_button.pack(pady=10)

image_label = tk.Label(root)
image_label.pack()

result_label = tk.Label(root, text="")
result_label.pack(pady=10)

# Start the GUI event loop
root.mainloop()
