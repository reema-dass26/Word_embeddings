import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

# Load the IMDb dataset
vocab_size = 10000
max_length = 200
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# Pad sequences to ensure uniform length
x_train = pad_sequences(x_train, maxlen=max_length, padding='post', truncating='post')
x_test = pad_sequences(x_test, maxlen=max_length, padding='post', truncating='post')


embedding_dim = 16

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train, epochs=10, batch_size=512, validation_split=0.2, verbose=1)

embedding_layer = model.layers[0]
embeddings = embedding_layer.get_weights()[0]

# Map indices to words
word_index = imdb.get_word_index()
index_to_word = {index + 3: word for word, index in word_index.items()}
index_to_word[0] = '<PAD>'
index_to_word[1] = '<START>'
index_to_word[2] = '<UNK>'
index_to_word[3] = '<UNUSED>'

# Map words to their embeddings
word_embeddings = {index_to_word[idx]: embeddings[idx] for idx in range(vocab_size) if idx in index_to_word}



import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Reduce dimensions for visualization
pca = PCA(n_components=2)
words = list(word_embeddings.keys())
vectors = list(word_embeddings.values())
reduced_embeddings = pca.fit_transform(vectors)

# Plot the embeddings
plt.figure(figsize=(15, 15))
for word, (x, y) in zip(words, reduced_embeddings):
    plt.scatter(x, y)
    plt.annotate(word, (x, y), fontsize=9)
plt.show()



vocab_size = len(word_index) + 1  # Adding 1 for padding token
embedding_dim = 8

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=padded_sequences.shape[1]),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(embedding_dim, activation='relu'),
    tf.keras.layers.Dense(vocab_size, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()



import numpy as np

# Create labels for the data (dummy labels for simplicity)
labels = np.array([0, 1, 2, 3])

# Train the model
model.fit(padded_sequences, labels, epochs=50)


embedding_layer = model.layers[0]
embeddings = embedding_layer.get_weights()[0]

# Map words to their embeddings
word_embeddings = {word: embeddings[idx] for word, idx in word_index.items()}



import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Reduce dimensions for visualization
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(list(word_embeddings.values()))

# Plot the embeddings
plt.figure(figsize=(10, 10))
for word, (x, y) in zip(word_index.keys(), reduced_embeddings):
    plt.scatter(x, y)
    plt.annotate(word, (x, y), fontsize=12)
plt.show()
