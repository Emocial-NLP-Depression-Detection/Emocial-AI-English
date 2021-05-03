import kerastuner
from tensorflow.keras.layers.experimental import preprocessing
from kerastuner.engine.hyperparameters import HyperParameter
from kerastuner.tuners import RandomSearch
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
import datetime
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.text import Tokenizer
from collections import Counter
from nltk.corpus import stopwords
import nltk
import string
import re
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import os
import time
print("\nStart reading dataset...\n")
# https://www.kaggle.com/c/nlp-getting-started : NLP Disaster Tweets

df = pd.read_csv("./data/sentiment_tweets3.csv")
print("Finish Reading\n")
# Preprocessing


def remove_URL(text):
    url = re.compile(r"https?://\S+|www\.\S+")
    return url.sub(r"", text)

# https://stackoverflow.com/questions/34293875/how-to-remove-punctuation-marks-from-a-string-in-python-3-x-using-translate/34294022


def remove_punct(text):
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)


pattern = re.compile(r"https?://(\S+|www)\.\S+")
for t in df.Tweets:
    matches = pattern.findall(t)
    for match in matches:
        print(t)
        print(match)
        print(pattern.sub(r"", t))
    if len(matches) > 0:
        break

print("\nStart cleaning dataset\n")
df["Tweets"] = df.Tweets.map(remove_URL)  # map(lambda x: remove_URL(x))
df["Tweets"] = df.Tweets.map(remove_punct)


# remove stopwords
# pip install nltk
nltk.download('stopwords')

# Stop Words: A stop word is a commonly used word (such as “the”, “a”, “an”, “in”) that a search engine
# has been programmed to ignore, both when indexing entries for searching and when retrieving them
# as the result of a search query.
stop = set(stopwords.words("english"))

# https://stackoverflow.com/questions/5486337/how-to-remove-stop-words-using-nltk-or-python


def remove_stopwords(text):
    filtered_words = [word.lower()
                      for word in text.split() if word.lower() not in stop]
    return " ".join(filtered_words)


df["Tweets"] = df.Tweets.map(remove_stopwords)
print("\ndone cleaning\n")

# Count unique words


def counter_word(text_col):
    count = Counter()
    for text in text_col.values:
        for word in text.split():
            count[word] += 1
    return count


counter = counter_word(df.Tweets)
num_unique_words = len(counter)

# Split dataset into training and validation set
train_size = int(df.shape[0] * 0.8)

train_df = df[:train_size]
val_df = df[train_size:]

# split text and labels
train_sentences = train_df.Tweets.to_numpy()
train_labels = train_df.label.to_numpy()
val_sentences = val_df.Tweets.to_numpy()
val_labels = val_df.label.to_numpy()

# Tokenize
print("\nNow Tokenizing...\n")
# vectorize a text corpus by turning each text into a sequence of integers
tokenizer = Tokenizer(num_words=num_unique_words)
tokenizer.fit_on_texts(train_sentences)  # fit only to training
word_index = tokenizer.word_index
train_sequences = tokenizer.texts_to_sequences(train_sentences)
val_sequences = tokenizer.texts_to_sequences(val_sentences)
print("\nDone Tokenizing\n")
NAME = "Emocial-LSTM-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir=os.path.join(
    "logs",
    "fit",
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
)
)

# Pad the sequences to have the same length
print("\nStart padding sequence\n")
# Max number of words in a sequence
max_length = 20

train_padded = pad_sequences(
    train_sequences, maxlen=max_length, padding="post", truncating="post")
val_padded = pad_sequences(
    val_sequences, maxlen=max_length, padding="post", truncating="post")
print(train_padded.shape, val_padded.shape)
print("\nDone padding sequence\n")

# flip (key, value)
reverse_word_index = dict([(idx, word) for (word, idx) in word_index.items()])


def decode(sequence):
    return " ".join([reverse_word_index.get(idx, "?") for idx in sequence])


# Create LSTM model
print("\nCreating model...\n")

LOG_DIR = f"tune/{int(time.time())}"


def build_model(hp):
    model = keras.models.Sequential()
    model.add(layers.Embedding(num_unique_words, 32, input_length=max_length))

# The layer will take as input an integer matrix of size (batch, input_length),
# and the largest integer (i.e. word index) in the input should be no larger than num_words (vocabulary size).
# Now model.output_shape is (None, input_length, 32), where `None` is the batch dimension.
    # CNN model
    model.add(layers.Conv1D(filters=hp.Choice('cnn filter', values=[32, 64], default=64), kernel_size=3, padding='same', activation=hp.Choice(
        'cnn_activation 1',
        values=['relu', 'tanh'],
        default='relu'
    )))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Dense(hp.Int("cnn dense 1", min_value=32,
              max_value=256, step=32), activation=hp.Choice(
        'dense_activation 1',
        values=['relu', 'tanh'],
        default='relu'
    )))
    # LSTM part
    model.add(layers.LSTM(hp.Int("input_units", min_value=32,
              max_value=256, step=32), dropout=0.1, return_sequences=True))
    model.add(layers.Dense(hp.Int("dense 1", min_value=32,
              max_value=256, step=32), activation=hp.Choice(
        'dense_activation 1',
        values=['relu', 'tanh'],
        default='relu'
    )))
    model.add(layers.Dense(1, activation="sigmoid"))
    # compile
    loss = keras.losses.BinaryCrossentropy(from_logits=False)
    optim = keras.optimizers.Adam(lr=hp.Float(
        'learning_rate',
        min_value=1e-3,
        max_value=1e-2,
        sampling='LOG',
        default=0.5e-2
    ))
    metrics = ["accuracy"]

    model.compile(loss=loss, optimizer=optim, metrics=metrics)
    return model


# model = build_model()
# model.fit(train_padded, train_labels, epochs=100, validation_data=(val_padded, val_labels), verbose=2)
tuner = RandomSearch(build_model, objective=kerastuner.Objective(
    "val_accuracy", direction="max"), max_trials=10, executions_per_trial=3, directory=LOG_DIR)
print(tuner.search_space_summary())
tuner.search(x=train_padded, y=train_labels, epochs=3,
             validation_data=(val_padded, val_labels))
print(tuner.results_summary())
# print(tuner.search_space_summary())
