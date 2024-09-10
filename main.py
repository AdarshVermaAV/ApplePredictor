import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dropout
import os

# Load and preprocess the dataset
def load_dataset(directory_path):
    images = []
    labels = []
    class_names = os.listdir(directory_path)
    class_to_index = {class_name: index for index, class_name in enumerate(class_names)}

    for class_name in class_names:
        class_dir = os.path.join(directory_path, class_name)
        if os.path.isdir(class_dir):
            for img_filename in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_filename)
                if img_path.lower().endswith(('png', 'jpg', 'jpeg')):
                    images.append(img_path)
                    labels.append(class_to_index[class_name])

    return images, np.array(labels), class_names

# Load dataset
image_dir = 'Apple_Dataset'
images, labels, class_names = load_dataset(image_dir)

# Convert labels to strings
labels_str = [str(label) for label in labels]

# Create a DataFrame for ImageDataGenerator
df = pd.DataFrame({
    'filename': images,
    'label': labels_str
})

# Create ImageDataGenerator with data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Create training data generator
train_gen = datagen.flow_from_dataframe(
    dataframe=df,
    x_col='filename',
    y_col='label',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Create validation data generator
val_gen = datagen.flow_from_dataframe(
    dataframe=df,
    x_col='filename',
    y_col='label',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Define and compile the model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Fine-tuning: Unfreeze some layers
base_model.trainable = True
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model = tf.keras.Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=[early_stopping]
)


def save_class_names(class_names, filename='class_names1.npy'):
    np.save(filename, np.array(class_names))

save_class_names(class_names)

model.save('apple_classifier_model1.h5')


loaded_model = tf.keras.models.load_model('apple_classifier_model1.h5')
