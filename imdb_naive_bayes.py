import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Load the IMDB dataset
(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.imdb.load_data(num_words=10000)

# Get the word index
word_index = tf.keras.datasets.imdb.get_word_index()

# Function to decode a review
def decode_review(text):
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

# Prepare the data
train_reviews = [decode_review(text) for text in train_data]
test_reviews = [decode_review(text) for text in test_data]

# Create a combined dataset for preprocessing
all_reviews = train_reviews + test_reviews

# Vectorize the reviews
vectorizer = CountVectorizer(stop_words=stopwords.words('english'))
all_vectors = vectorizer.fit_transform(all_reviews)

# Split the combined dataset back into training and testing sets
train_vectors = all_vectors[:len(train_reviews), :]
test_vectors = all_vectors[len(train_reviews):, :]

# Train the model
clf = MultinomialNB()
clf.fit(train_vectors, train_labels)

# Test the model
predictions = clf.predict(test_vectors)

# Calculate the accuracy
accuracy = accuracy_score(test_labels, predictions)
print(f'Test accuracy: {accuracy}')
