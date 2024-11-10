#%% 1.1. Import modules
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import tf_keras
import pickle
import os
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
#%% 1.2. Load the Cassava dataset (TensorFlow dataset)
# NOTE: It can take a lot of time (I have tested and it's about 10-20')!!!
# tensorflow_datasets is saved in C:\Users\Admin\tensorflow_datasets
(ds_train, ds_validation, ds_test), ds_info = tfds.load(
    name='cassava',
    split=['train', 'validation', 'test'],
    with_info=True,
    as_supervised=True
)
#%% 1.3. Datasets informations
'''Cassava consists of leaf images for the cassava plant depicting 
1.healthy and four (4) disease conditions:
2.Cassava Mosaic Disease (CMD) 
3.Cassava Bacterial Blight (CBB) 
4.Cassava Greem Mite (CGM)
5.Cassava Brown Streak Disease (CBSD)'''

print(f"Number of training examples: {ds_info.splits['train'].num_examples}")
print(f"Number of validation examples: {ds_info.splits['validation'].num_examples}")
print(f"Number of test examples: {ds_info.splits['test'].num_examples}")
print(f"Number of classes: {ds_info.features['label'].num_classes}")
print(f"Class names: {ds_info.features['label'].names}")

#%% 1.4. Preprocessing datasets
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
#%% 2.1. Create the model
def create_model():
    model_url = "https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5"
    base_model = hub.KerasLayer(model_url, trainable=False)

    num_classes = ds_info.features['label'].num_classes

    model = tf_keras.Sequential([
        base_model,
        tf_keras.layers.Dense(128, activation='relu'),
        tf_keras.layers.Dropout(0.5),
        tf_keras.layers.Dense(num_classes, activation='softmax')
    ])

    return model
#%% 2.2. Learning rate schedule
def lr_schedule(epoch):
    lr = 0.001
    if epoch > 5:
        lr *= 0.1
    return lr

# 2.3. Implement k-fold cross-validation
def k_fold_cross_validation(k=5):
    # Get the total number of examples in the dataset
    total_examples = tf.data.experimental.cardinality(ds_train).numpy()
    
    # Create index array
    indices = np.arange(total_examples)
    
    # Initialize KFold
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    
    all_scores = []
    for fold, (train_indices, val_indices) in enumerate(kfold.split(indices)):
        print(f"Fold {fold + 1}/{k}")
        
        # Create datasets for this fold
        train_data = ds_train.take(train_indices.size).shuffle(buffer_size=train_indices.size)
        val_data = ds_train.take(val_indices.size).shuffle(buffer_size=val_indices.size)
        
        # Prepare datasets
        train_data = prepare_dataset(train_data, train=True)
        val_data = prepare_dataset(val_data)
        
        model = create_model()
        model.compile(
            optimizer=tf_keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        
        # Train the model
        history = model.fit(
            train_data,
            epochs=10,
            validation_data=val_data,
            callbacks=[
                tf_keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
                tf_keras.callbacks.LearningRateScheduler(lr_schedule)
            ]
        )
        
        # Evaluate the model
        scores = model.evaluate(val_data)
        all_scores.append(scores[1])  # Append accuracy
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'Fold {fold + 1} - Training and Validation Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Fold {fold + 1} - Training and Validation Loss')
        plt.legend()
        plt.show()
    
    print(f"Average accuracy across all folds: {np.mean(all_scores):.4f}")
# 2.3. Run k-fold cross-validation
k_fold_cross_validation()

#%% 2.4. Train the final model
# NOTE: It also takes a lot of time!!!
model = create_model()
model.compile(
    optimizer=tf_keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[
        tf_keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf_keras.callbacks.LearningRateScheduler(lr_schedule)
    ]
)
#%% 3.1. Evaluate the model
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test accuracy: {test_accuracy:.2f}")

# 3.2. Plot training history
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

# 3.3. Save the model
model.save('plant_disease_model.h5')
print("Model saved as 'plant_disease_model.h5'")

# 3.4. Function to predict on a single image
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

# 3.5. Visualize sample images and their predictions
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

# 3.6. Plot confusion matrix
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
#%% 4.1. GUI implementation
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

# 4.2. Create the main window
root = tk.Tk()
root.title("Plant Disease Classifier")

# 4.3. Create and pack widgets
open_button = tk.Button(root, text="Open Image", command=open_image)
open_button.pack(pady=10)

image_label = tk.Label(root)
image_label.pack()

result_label = tk.Label(root, text="")
result_label.pack(pady=10)

# 4.4. Start the GUI event loop
root.mainloop()

# %%
