# Tweet-Decoder-PPOC
# Tweet Decode — Sentiment Analysis of Political Tweets

**Tweet Decode** is a sentiment analysis project built using Python, focusing on tweets related to the 2019 Indian Lok Sabha elections. The project aims to extract and compare public sentiment toward major political figures using natural language processing (NLP) techniques.

## Project Overview

This project classifies tweets as **positive**, **negative**, or **neutral** using TextBlob. The analysis was performed on two datasets containing over 39,000 tweets related to Narendra Modi and Rahul Gandhi.

### Key Features:
- Data cleaning and preprocessing using `pandas` and `NumPy`
- Sentiment classification using `TextBlob`
- Balanced comparison using randomly selected tweets
- Visualization of sentiment distribution via `matplotlib`, `seaborn`, and word clouds
- Real-world application in political sentiment tracking

## Tools & Libraries Used
- `pandas`, `NumPy` for data manipulation
- `TextBlob` for sentiment analysis
- `matplotlib`, `seaborn`, `plotly` for visualization
- `wordcloud` for visual sentiment representation

## Results
- Modi: ~64% Positive, ~26% Negative tweets
- Rahul: ~59.6% Positive, ~30.4% Negative tweets
- Word clouds revealed common political keywords and hashtags

## Future Work
- Incorporate more diverse data sources
- Use advanced NLP models like BERT or Vader
- Extend analysis to real-time political events
