# Regression & Tweet Classification with TensorFlow

This repository contains a set of machine learning and natural language processing (NLP) tasks completed as part of a foundational course. The aim was to understand regression from scratch and build NLP-based classifiers using TensorFlow and pretrained models.

##  Project Overview

###  Assignment 1: Linear Regression from Scratch
- Built a simple linear regression model using NumPy only.
- Predicted car prices based on numerical features.
- Implemented gradient descent, weight updates, and error metrics (MSE, R²) from scratch.
- Achieved high R² accuracy on a normalized car dataset.

###  Assignment 2: Feedforward Neural Network (TensorFlow)
- Generated a regression dataset using `sklearn.datasets.make_regression`.
- Built a fully connected neural network using the TensorFlow Sequential API.
- Used Mean Squared Error as loss and SGD optimizer with learning rate tuning.
- Visualized loss curves and evaluated model accuracy using R².

###  Assignment 3: Tweet Sentiment Classification with BERT
- Built a binary classification model using the Universal Sentence Encoder (USE) from TensorFlow Hub.
- Classified political tweets from the IMDb/tweet-like dataset as positive or negative.
- Implemented data preprocessing, embedding, training, and evaluation using a custom neural network.
- Visualized model loss and accuracy trends across epochs.

###  Assignment 4: Political Tweet Categorization (Pro-Gov vs Opposition)
- Processed a dataset of 49,000+ real-world political tweets.
- Embedded tweets using the Universal Sentence Encoder (512D vector representation).
- Built and trained a deep neural network with multiple hidden layers using TensorFlow.
- Achieved **78% accuracy** in classifying tweets into **pro-government** vs **opposition** categories.
- Evaluated model with precision, recall, and F1-score.

##  Technologies Used
- Python
- NumPy, pandas, matplotlib
- Scikit-learn
- TensorFlow & Keras
- TensorFlow Hub (BERT)

##  Results
- Regression model implemented from scratch and evaluated using custom MSE and R² functions.
- Fully connected neural network trained to fit synthetic data.
- NLP sentiment classification using pretrained embeddings.
- Tweet classification model achieved 78% accuracy on unseen data.


