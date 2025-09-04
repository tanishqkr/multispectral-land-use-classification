import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

# --- Configuration ---
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20
DATA_DIR = 'data'
# The CLASSES list is crucial now to explicitly define your new categories.
CLASSES = ['AnnualCrop', 'Industrial', 'Pasture', 'Residential', 'SeaLake', 'Highway', 'River'] 
NUM_CLASSES = len(CLASSES)
MODEL_PATH = 'models/single_model.h5'

# --- 1. Prepare Data ---
print("Loading data...")
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    classes=CLASSES  # This ensures the generator only loads the correct folders.
)

validation_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    classes=CLASSES  # This ensures the generator only loads the correct folders.
)

# Compute class weights to handle imbalance
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights_dict = dict(zip(np.unique(train_generator.classes), class_weights))
print("Computed class weights:", class_weights_dict)

# --- 2. Build and Train the Model ---
print("Building and training the single model...")
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    class_weight=class_weights_dict
)

# Save the model
if not os.path.exists('models'):
    os.makedirs('models')
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
print("\nTraining complete.")