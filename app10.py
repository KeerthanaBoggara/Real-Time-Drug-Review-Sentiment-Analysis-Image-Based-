import os
import random
import numpy as np
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM, Reshape
from tensorflow.keras.callbacks import ModelCheckpoint
from PIL import Image
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", message="missing ScriptRunContext!")

# Directory Paths
model_path = r"C:/Flask/model.h5"  # Text sentiment model path
image_model_path = r"C:/Flask/cnn_lstm_model.h5"  # CNN+LSTM image model path
image_dir = r"C:/Flask/dataset"  # Folder containing sentiment images

# Sentiment Labels
sentiment_labels = {2: "positive", 1: "neutral", 0: "negative"}

# Function to create CNN + LSTM model
def create_cnn_lstm_model(input_shape=(48, 48, 3)):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, name="conv2d_1"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', name="conv2d_2"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', name="conv2d_3"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Reshape((1, -1)))  # Reshape for LSTM
    model.add(LSTM(128, return_sequences=False))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load the text-based model for sentiment analysis
def load_text_model():
    try:
        return load_model(model_path)
    except Exception as e:
        st.write(f"Error loading text model: {e}")
        return None

# Function to train the CNN + LSTM model
# Function to train the CNN + LSTM model and return history
def train_image_model(epochs=20):
    image_model = create_cnn_lstm_model()
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        image_dir, target_size=(48, 48), batch_size=32, class_mode='categorical', shuffle=True
    )
    
    checkpoint = ModelCheckpoint(image_model_path, save_best_only=True)
    history = image_model.fit(train_generator, epochs=epochs, steps_per_epoch=train_generator.samples // 32, callbacks=[checkpoint])

    return image_model, history  # Returning history for accuracy plot


# Load the text model
text_model = load_text_model()

# Streamlit UI
st.title("Drug Review Sentiment Analysis")
user_input = st.text_area("Enter your review or a number (0: Negative, 1: Neutral, 2: Positive)")
epochs_input = st.number_input("Enter number of epochs for model training", min_value=1, max_value=100, value=20)

# Button to train the model
if st.button("Train Image Model"):
    if text_model is not None:
        st.write("Training image model with CNN + LSTM...")
        image_model, history = train_image_model(epochs=epochs_input)
        st.write(f"Model trained for {epochs_input} epochs!")
        st.line_chart(history.history['accuracy'])
        st.line_chart(history.history['loss'])
    else:
        st.write("Text model not loaded. Cannot proceed with image model training.")

# Button to predict sentiment
if st.button("Predict"):
    sentiment = None
    
    try:
        user_input = int(user_input)
        sentiment = user_input if user_input in [0, 1, 2] else None
    except ValueError:
        tokenizer = Tokenizer()
        sequence = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(sequence, maxlen=200)
        sentiment = np.argmax(text_model.predict(padded))
    
    # Check if sentiment is valid
    if sentiment is not None:
        st.write(f"Predicted Sentiment: {sentiment_labels[sentiment]}")

        # Get the correct sentiment folder
        sentiment_folder = os.path.join(image_dir, sentiment_labels[sentiment])

        if os.path.isdir(sentiment_folder):
            images = [f for f in os.listdir(sentiment_folder) if f.endswith(('.jpg', '.png'))]
            if images:
                random_image = random.choice(images)
                img_path = os.path.join(sentiment_folder, random_image)
            else:
                img_path = None
        else:
            img_path = None

        # Display image if found
        if img_path and os.path.exists(img_path):
            image = Image.open(img_path)
            st.image(image, caption=sentiment_labels[sentiment], use_container_width=True)
        else:
            st.write("No image found for this sentiment.")
    
    else:
        st.write("Invalid input. Enter a review or a valid number (0-2).")

def plot_accuracy(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history.get('val_accuracy', []), label='Validation Accuracy', linestyle='dashed')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('CNN + LSTM Model Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy_plot.png")  # Save the accuracy diagram
    plt.show()

