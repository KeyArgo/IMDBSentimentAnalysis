<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/KeyArgo/IMDBSentimentAnalysis">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>
<h2 align="center">IMDBSentimentAnalysis using Naive Bayes</h2>

  <p align="center">
    <a href="https://github.com/KeyArgo/ImageClassificationCNN"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/KeyArgo/ImageClassificationCNN">View Demo</a>
    ·
    <a href="https://github.com/KeyArgo/ImageClassificationCNN/issues">Report Bug</a>
    ·
    <a href="https://github.com/KeyArgo/ImageClassificationCNN/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This repository contains an implementation of a Naive Bayes classifier for sentiment analysis on the IMDB movie reviews dataset using the Scikit-learn and TensorFlow libraries.


<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.


# IMDB Sentiment Analysis using Naive Bayes

This repository contains a TensorFlow-based implementation of a Convolutional Neural Network (CNN) for image classification on the CIFAR-10 dataset.

This repository contains an implementation of a Naive Bayes classifier for sentiment analysis on the IMDB movie reviews dataset using the Scikit-learn and TensorFlow libraries.

## Dependencies

To run this code, you need to have the following packages installed:

- Numpy
- TensorFlow
- Scikit-learn
- NLTK

You can install these packages using pip:

```bash
pip install numpy tensorflow scikit-learn nltk

```

## Dataset

The IMDB movie reviews dataset consists of 50,000 movie reviews, with 25,000 for training and 25,000 for testing. Each review is labeled as either positive (1) or negative (0), indicating the sentiment of the reviewer.

## Naive Bayes Classifier

The classifier is implemented using the Multinomial Naive Bayes algorithm from the Scikit-learn library. The text reviews are preprocessed using the CountVectorizer, which vectorizes the reviews and removes common English stopwords using the NLTK library.

## Training and Evaluation

The classifier is trained on the training dataset and evaluated on the test dataset. The test accuracy is calculated and printed.

## Usage

To train and evaluate the model, simply run the provided code in a Python environment with the required dependencies installed.

```bash
python imdb_naive_bayes.py
```
This will train the model and print the test accuracy upon completion.

This will train the model and print the test accuracy upon completion.


## Preprocessing Steps

1. Decode the reviews using the IMDB dataset's word index.
2. Merge the training and test datasets to create a combined dataset for preprocessing.
3. Vectorize the reviews using CountVectorizer and remove common English stopwords using the NLTK library.
4. Split the combined dataset back into training and testing sets.

## Code Overview

1. Import the necessary libraries (NumPy, TensorFlow, Scikit-learn, NLTK).
2. Load the IMDB dataset and word index.
3. Define a function to decode a review using the word index.
4. Prepare the data by decoding the reviews, merging the datasets, and vectorizing the text.
5. Train a Multinomial Naive Bayes classifier on the training dataset.
6. Test the model on the test dataset and calculate the test accuracy.
7. Print the test accuracy.

## Dependencies

To run this code, you need to have the following packages installed:

- TensorFlow (2.x)
- NumPy
- Scikit-learn
- NLTK

You can install these packages using pip:

```bash
pip install tensorflow numpy scikit-learn nltk
```

## Dataset

The dataset used for this project is the IMDB movie review dataset. It contains 50,000 movie reviews split into a training set (25,000 reviews) and a testing set (25,000 reviews). Each review is labeled as positive (1) or negative (0).

## Naive Bayes Classifier

The model used for classification is the Multinomial Naive Bayes classifier from the Scikit-learn library. This classifier is suitable for text classification tasks, especially when the dataset is large and sparse.

## Usage

To train and evaluate the model, simply run the provided code in a Python environment with the required dependencies installed.

```python
python imdb_naive_bayes.py
```

Performance
The Multinomial Naive Bayes classifier is expected to perform well on the IMDB movie review dataset due to its simplicity and suitability for text classification tasks. The performance may vary depending on the specific dataset used and the preprocessing techniques applied.


## Preprocessing

The dataset is preprocessed using the following steps:

1. Decoding the IMDB reviews from integer sequences back to text.
2. Combining the training and testing sets to create a single dataset for preprocessing.
3. Vectorizing the reviews using the `CountVectorizer` from Scikit-learn library. This converts the text data into a bag-of-words representation.
4. Removing English stopwords using the Natural Language Toolkit (NLTK).
5. Splitting the combined dataset back into training and testing sets.


## License

This project is licensed under the MIT License. You are free to use, modify, and distribute the code as long as the original copyright and permission notice are included.

## Contributing

If you have any suggestions or improvements for this project, feel free to contribute. You can fork the repository, make your changes, and submit a pull request. We appreciate your contributions and will review them as soon as possible.

### Steps for contributing:

1. Fork the repository on GitHub.
2. Clone your fork of the repository: `git clone https://github.com/KeyArgo/IMDBSentimentAnalysis.git`
3. Create a new branch for your changes: `git checkout -b my-feature-branch`
4. Make your changes to the code or documentation.
5. Commit your changes: `git commit -am 'Add my new feature'`
6. Push your changes to your fork: `git push origin my-feature-branch`
7. Create a new pull request on the original repository.

For any issues or questions, please open an issue on the GitHub repository.

## Acknowledgements

The dataset used in this project is provided by the TensorFlow library and is originally from the [IMDB Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/). The implementation is based on the Multinomial Naive Bayes classifier from the Scikit-learn library and preprocessing techniques from the NLTK library.

## Daniel LaForce

Daniel LaForce - https://github.com/KeyArgo

Please feel free to reach out with any questions, suggestions, or feedback.


```python
python cifar10_cnn.py
```

This will train the model and print the test accuracy upon completion.

## Contributing

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

## Contact

Daniel LaForce - danlaforce3@gmail.com

Project Link: [https://github.com/KeyArgo/IMDBSentimentAnalysis](https://github.com/KeyArgo/IMDBSentimentAnalysis)
