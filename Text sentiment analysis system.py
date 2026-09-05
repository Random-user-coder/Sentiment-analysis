import re
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
from keras_visualizer import visualizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential  
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
import pickle
import tkinter as tk
from tensorflow import keras
import tkinter as tk
from tkinter import messagebox
import os

# Loading the "Reviews" dataset and storing it in the variable df_amazon
df_amazon = pd.read_csv("Reviews.csv", on_bad_lines='skip', engine="python")
# keeping the column'Score' and 'Text' and removing missing rows in them
df_amazon = df_amazon[['Score', 'Text']].dropna()

# Defining the ratings and replace the sentiment with the labels we have defined
labels = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}
df_amazon['sentiment']= df_amazon['Score'].map(labels)

# Creating a function to clean out any unnecessary characters in the text
def clean_the_text(text):
    text = re.sub(r'https?:www\S+', '', text) #revomes URL
    text = re.sub(r'[^\w\s]','', text) #removes punctuation
    text = re.sub(r'\s+', ' ',text) #cleans up extra whitespace
    text = re.sub("\d+",'' , text ) #removes digits
    text = re.sub('[^a-zA-Z\s]','', text) #removes special characters
    text = text.lower() #coverts the text to lowercase
    return text
df_amazon['Text'] = df_amazon['Text'].apply(clean_the_text)

# Now we are tokenizing the texts, which basically means converting words into numbers
vocab_size = 20000
max_length = 50


tokenizer = Tokenizer(num_words = vocab_size, oov_token= "<OOV>")# Adjusting the maximum number of words and defining the OOV token
tokenizer.fit_on_texts(df_amazon['Text']) # Convert words in the 'Text' column into numbers
sequences = tokenizer.texts_to_sequences(df_amazon['Text'])
# Padding our sequences so they are all the same length
padding_sequences = tf.keras.utils.pad_sequences(
    sequences = sequences ,
    maxlen = max_length,
    dtype='int32',
    padding='post',
    truncating='post')
# Converting our sentiment labels into numpy arrays
label= np.array(df_amazon['sentiment'])
# Saving both the input padded sequences and the sentiment labels as npy files
np.save('padding_sequences.npy', padding_sequences)
np.save('sentiment.npy', label)


# Defining our X value and y value for model training
X = np.load('padding_sequences.npy')
y = np.load('sentiment.npy')

if not os.path.exists("model.h5"):
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
    # Building our LSTM model
    model = keras.Sequential(
    [keras.layers.Embedding(vocab_size, output_dim=100, input_length=max_length), # The first layer converts all the words to a fixed size
    keras.layers.SpatialDropout1D(rate = 0.2), # Helps our model to not overfit
    keras.layers.LSTM(128),# 128 units to learn patterns in the sequence
    keras.layers.Dense(3, activation='softmax')]) # Lastly the text is put in one of these 3 catogories (postive,negative,neutral)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
    metrics=['accuracy'])
    print("model successfully compiled")

    history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val))

    model.save("model.h5")

# Saving the tokenizer
import pickle
with open("amazon_tokenizer.pickle", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Loading the model
model = load_model("model.h5")

# Then loading the tokenizer
with open("amazon_tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

# Lastly defining the sentiment labels
sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive' }

# Creating a function to predict the sentiment of the inputed text
def predict_sentiment(text):
    clean_text = clean_the_text(text) # Clean the text using our cleaning function
    sequence = tokenizer.texts_to_sequences([clean_text]) # Converts all the words to numbers
    padded_sequence = pad_sequences(sequence, maxlen=50) # Adding the padded sequence to make it in to a fixed number of length
    prediction = model.predict(padded_sequence) # Then by using the trained model to predict the inputed text
    sentiment_index = np.argmax(prediction) # Get the highest prediction from the model
    sentiment_label = sentiment_map.get(sentiment_index) # Then coverting the sentiment index into the sentiment label

    # Extracting the sentiments from prediction arrays
    negative_percentage = int(prediction[0][0] * 100)
    neutral_percentage = int(prediction[0][1] * 100)
    positive_percentage = int(prediction[0][2] * 100)

    return {
        "Sentiment": sentiment_label,
        "Negative": negative_percentage,
        "Neutral": neutral_percentage,
        "Positive": positive_percentage}

def run_gui():
    def on_predict():
        review = entry.get("1.0", tk.END).strip()
        if not review:
            messagebox.showerror("Input Error", "Please enter a review.")
            return
        predicted_sentiment = predict_sentiment(review)
        output_var.set(f"Predicted: {predicted_sentiment['Sentiment']}\n\n"
                       f"Class Probabilities:\n"
                       f"Negative: {predicted_sentiment['Negative']}%\n"
                       f"Neutral: {predicted_sentiment['Neutral']}%\n"
                       f"Positive: {predicted_sentiment['Positive']}%")

    root = tk.Tk()
    root.title("Amazon Sentiment Classifier")
    root.geometry("450x400")

    tk.Label(root, text="Enter Review:", font=("Arial", 12)).pack(pady=5)
    entry = tk.Text(root, height=5, width=50)
    entry.pack(pady=5)

    tk.Button(root, text="Predict Sentiment", command=on_predict).pack(pady=10)

    output_var = tk.StringVar()
    tk.Label(root, textvariable=output_var, font=("Arial", 11), fg="blue", wraplength=400).pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    run_gui()