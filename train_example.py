import pandas as pd
import numpy as np
import os
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# Step 1: Prepare train, validation, and test datasets
train_data_dir = 'path_to_your_train_data_directory'
val_data_dir = 'path_to_your_validation_data_directory'
test_data_dir = 'path_to_your_test_data_directory'

img_width, img_height = 224, 224
batch_size = 32

# Load the DataFrame containing image paths and labels
df_train = pd.read_csv('path_to_train_data_labels.csv')
df_val = pd.read_csv('path_to_validation_data_labels.csv')
df_test = pd.read_csv('path_to_test_data_labels.csv')

# Step 1 (continued): Define a function to load images from paths and corresponding labels
def load_images_from_paths(image_paths, labels):
    images = []
    for path in image_paths:
        img = load_img(path, target_size=(img_width, img_height))
        img = img_to_array(img)
        images.append(img)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

# Create the data generators with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    zoom_range=(0.73, 0.9),
    horizontal_flip=True,
    rotation_range=10,
    width_shift_range=0.10,
    fill_mode='constant',
    height_shift_range=0.10,
    brightness_range=(0.55, 0.9)
)

valid_test_datagen = ImageDataGenerator(rescale=1./255)

x_train, y_train = load_images_from_paths(df_train['image_path'].values, df_train['label'].values)
x_valid, y_valid = load_images_from_paths(df_val['image_path'].values, df_val['label'].values)
x_test, y_test = load_images_from_paths(df_test['image_path'].values, df_test['label'].values)

train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size, shuffle=True)
valid_generator = valid_test_datagen.flow(x_valid, y_valid, batch_size=batch_size, shuffle=True)
test_generator = valid_test_datagen.flow(x_test, y_test, batch_size=batch_size, shuffle=False)

# Step 2: Load model weights
original_model = keras.models.load_model('original_model.h5')

# Step 3: Change backbone for transfer learning
binary_model = keras.models.Sequential()
for layer in original_model.layers[:-1]:
    binary_model.add(layer)
binary_model.add(keras.layers.Dense(1, activation='sigmoid'))

# Step 4: Hyperparameter tuning with validation dataset
binary_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
binary_model.fit(train_generator, epochs=10, validation_data=valid_generator)

# Step 5: Choose the final model
final_model = binary_model

# Step 6: Test on the testing dataset
test_loss, test_acc = final_model.evaluate(test_generator)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)
