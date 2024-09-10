import os
import cv2
import random
import numpy as np
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

def load_and_preprocess_images(categories, dataset_path):
    data = []
    for category in categories:
        path = os.path.join(dataset_path, category)
        label = categories.index(category)
        for file in os.listdir(path):
            img_path = os.path.join(path, file)
            img = cv2.imread(img_path)
            if img is not None:  # Ensure image is loaded correctly
                img = cv2.resize(img, (224, 224))
                img = preprocess_input(img)  # Preprocess for VGG16
                data.append([img, label])
    return data

def main():
    # Define categories and dataset path
    categories = ['GalaApples', 'RoyalApple', 'GoldenApples', 'FujiApples', 'GrannySmith', 'NewtonPipping', 'EmpireApple']
    dataset_path = 'Apple_Dataset'

    # Load and preprocess images
    data = load_and_preprocess_images(categories, dataset_path)

    # Shuffle and split data
    random.shuffle(data)
    x, y = zip(*data)  # Unzip data
    x = np.array(x)
    y = np.array(y)

    # Convert labels to categorical format
    y = to_categorical(y, num_classes=len(categories))

    # Split into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Build the model
    vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = Sequential([
        vgg,
        GlobalAveragePooling2D(),
        Dense(len(categories), activation='softmax')
    ])

    # Freeze VGG16 layers
    for layer in vgg.layers:
        layer.trainable = False

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    # Train the model with data augmentation
    model.fit(
        datagen.flow(X_train, Y_train, batch_size=32),
        epochs=50,  # You can increase this number
        validation_data=(X_test, Y_test),
        callbacks=[early_stopping]
    )

    # Save the model and class names
    model.save('apple_classifier_vgg16.h5')
    np.save('class_names.npy', np.array(categories))

if __name__ == "__main__":
    main()
