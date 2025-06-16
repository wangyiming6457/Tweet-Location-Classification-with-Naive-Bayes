# 🧠 Tweet Location Classification with Naive Bayes

This project uses a **Naive Bayes Classifier** to predict the geographic origin of tweets—specifically from **New York**, **London**, or **Paris**—based on the tweet's text content.

## 🌍 Project Objective

Using text data from real tweets, the goal is to classify which city a tweet likely came from. By training a model on word usage patterns, we can infer regional language differences using a probabilistic classifier.

## 📝 Dataset

The project uses three JSON files:
- `new_york.json`
- `london.json`
- `paris.json`

Each file contains tweets from the respective city, stored as JSON lines.

## 🛠️ Feature Preparation

The tweets are first extracted into lists:
```python
new_york_text = new_york_tweets["text"].tolist()
london_text = london_tweets["text"].tolist()
paris_text = paris_tweets["text"].tolist()
