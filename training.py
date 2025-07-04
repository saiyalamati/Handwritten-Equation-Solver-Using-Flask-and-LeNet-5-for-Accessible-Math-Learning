import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K

# Load dataset
df_train = pd.read_csv("dataset.csv", index_col=False)

# Ensure last column is the label
labels = df_train.iloc[:, -1]

# Remove labels from feature data
df_train.drop(df_train.columns[-1], axis=1, inplace=True)

# Convert labels to integers and handle potential non-numeric values
labels = pd.to_numeric(labels, errors="coerce").fillna(0).astype(int)

# Ensure labels are within a valid ran
num_classes = labels.nunique()
if labels.min() < 0 or labels.max() >= num_classes:
    raise ValueError("Label values must be between 0 and the number of classes - 1.")

# Convert labels to categorical (one-hot encoding)
cat = to_categorical(labels, num_classes)

# Convert training data to NumPy array and reshape for CNN
x_train = df_train.to_numpy().astype("float32")
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)  # (samples, height, width, channels)

# Normalize pixel values to range [0,1]
x_train /= 255.0

# Set image format to "channels_last" for TensorFlow compatibility
K.set_image_data_format("channels_last")

# Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")
])

# Compile model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train model
model.fit(x_train, cat, epochs=10, batch_size=200, verbose=1)

# Save model architecture to JSON
model_json = model.to_json()
with open("model1.json", "w") as json_file:
    json_file.write(model_json)

# Save model weights
model.save_weights("model1.weights.h5")

print("Model training completed and saved successfully.")
