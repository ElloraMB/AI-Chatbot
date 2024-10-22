import json
import nltk
import random
import pickle
import numpy as np
import tensorflow as tf


from nltk.stem import WordNetLemmatizer
from tensorflow import keras
from keras.models import Sequential # type: ignore
from keras.layers import Dense, Activation, Dropout # type: ignore
from keras.optimizers import SGD # type: ignore
from keras.preprocessing.sequence import pad_sequences # type: ignore

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', ',', '!', '.']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        documents.append((word_list, intent['tag']))
        words.extend(word_list)  # Append words to the 'words' list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

print("Words:", words)
print("Classes:", classes)

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    # Create a bag of words with 1s and 0s
    bag = [1 if word in word_patterns else 0 for word in words]

    # Create the output row
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1

    # Append the bag and output_row as separate lists
    training.append([bag, output_row])

# Shuffle the training data
random.shuffle(training)

# Split the training data into X and Y
train_x = np.array([item[0] for item in training])
train_y = np.array([item[1] for item in training])

print("Train X shape:", train_x.shape)
print("Train Y shape:", train_y.shape)

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

print(model.summary())

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
print("done")
