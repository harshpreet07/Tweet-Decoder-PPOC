
## Tweet Classification and Regression using Machine Learning

### Project Overview

This project contains a series of machine learning assignments that build progressively from basic regression models to advanced deep learning-based tweet classification. The final objective is to predict political sentiment (pro-government or opposition) from tweets using TensorFlow and pre-trained NLP embeddings.
<br/>

### Assignments Breakdown

<br/>

#### Assignment 1: Linear Regression from Scratch

* **Goal:** Predict car prices using a self-implemented linear regression model.
* **Skills:** Gradient Descent, Cost Function, Manual Weight Updates
* **Tech:** NumPy, Pandas, Matplotlib
* **Evaluation:** Mean Squared Error (MSE), R² Score

<br/>

#### Assignment 2: Regression using TensorFlow

* **Goal:** Predict a continuous target using a feedforward neural network.
* **Model:** `Dense(50) → Dense(10) → Dense(5) → Dense(1)`
* **Embedding/Features:** Synthetic 10-feature dataset using `make_regression`
* **Tech:** TensorFlow (Sequential API), SGD optimizer, Loss Plotting

<br/>

#### Assignment 3: Sentiment Regression on Tweets

* **Goal:** Predict sentiment polarity score from preprocessed tweets.
* **Embedding:** Universal Sentence Encoder (USE) from TensorFlow Hub
* **Model:** Multi-layer DNN using ReLU activations
* **Evaluation:** MSE and R² Score

<br/>

#### Assignment 4: Tweet Classification (Final Task)

* **Goal:** Classify tweets into `pro-government (1)` or `opposition (0)`
* **Data:** 49,477 pre-labeled tweets
* **Embedding:** USE converts each tweet into 512-dimensional semantic vectors
* **Model:**
<br/>
  ```text
  Normalization → Dense(150) → Dense(50) → Dense(18) → Dense(6) → Dense(2, softmax)
  ```
* **Loss Function:** Sparse Categorical Crossentropy
* **Optimizer:** Adam
* **Evaluation:** Accuracy, Precision, Recall, F1-score

<br/>

### Results (Final Model)

* **Validation Accuracy:** \~77%
* **Precision & Recall:** Balanced across both classes
* **Use-case:** Political sentiment classification from short text (tweets)

<br/>

### Libraries Used

* `NumPy`, `Pandas`, `Matplotlib`
* `TensorFlow`, `TensorFlow Hub`
* `Scikit-learn`

<br/>
###  Project Highlights

* Built linear regression from scratch using only NumPy
* Embedded natural language using pre-trained Universal Sentence Encoder
* Designed and evaluated deep learning models for both regression and classification
* Achieved \~78% accuracy in real-world political tweet classification
<br/>
