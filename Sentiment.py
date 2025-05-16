from google.colab import drive
drive.mount('/content/drive')

import os
import keras
import numpy as np
import matplotlib.pyplot as plt
# from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D
from keras.applications import VGG16, ResNet50, InceptionV3
from keras.optimizers import Adam
import random
import seaborn as sns
from skimage import io
from skimage.transform import resize
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Set the path to your data folder
data_dir = '/content/drive/MyDrive/ResearchPaper'
class_names = ['NonCataract', 'Cataract']

# Set hyperparameters
batch_size = 32
epochs = 10
input_shape = (100, 100, 3)  # Adjust image size as needed

# Data augmentation for training set
train_datagen = ImageDataGenerator(rescale=1.0 / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

# Validation and test data should only be rescaled (no augmentation)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Data generators
train_generator = train_datagen.flow_from_directory(data_dir,
                                                    target_size=input_shape[:2],
                                                    batch_size=batch_size,
                                                    class_mode='binary',
                                                    classes=class_names)

val_generator = val_datagen.flow_from_directory(data_dir,
                                                target_size=input_shape[:2],
                                                batch_size=batch_size,
                                                class_mode='binary',
                                                classes=class_names,
                                                shuffle=False)

test_generator = test_datagen.flow_from_directory(data_dir,
                                                  target_size=input_shape[:2],
                                                  batch_size=batch_size,
                                                  class_mode='binary',
                                                  classes=class_names,
                                                  shuffle=False)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples // batch_size,
                    epochs=epochs,
                    validation_data=val_generator,
                    validation_steps=val_generator.samples // batch_size)

# Plot test and validation accuracy per epoch
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot test and validation loss per epoch
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
print("Test Accuracy:", test_accuracy)
print("Test Loss:", test_loss)

# Get the true labels and predicted probabilities for the test set
y_true = test_generator.classes
y_probs = model.predict(test_generator).ravel()

# Calculate AUC score
auc_score = roc_auc_score(y_true, y_probs)
print("AUC Score:", auc_score)

# Generate the classification report and confusion matrix
y_pred = y_probs > 0.5
print("Classification Report:")
print(classification_report(y_true, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# Plot the AUC curve
fpr, tpr, thresholds = roc_curve(y_true, y_probs)
plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


