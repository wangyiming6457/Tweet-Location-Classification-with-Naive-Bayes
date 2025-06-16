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

all_tweets = new_york_text + london_text + paris_text
labels = [0] * len(new_york_text) + [1] * len(london_text) + [2] * len(paris_text)
```

Where:

- `0` = New York
- `1` = London
- `2` = Paris

## ✂️ Train-Test Split

Split data into training and testing sets:

```python
from sklearn.model_selection import train_test_split

train_data, test_data, train_labels, test_labels = train_test_split(
    all_tweets, labels, test_size=0.2, random_state=1
)
```

## 🔠 Text Vectorization

Convert tweets into numeric vectors using **Bag-of-Words**:

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
train_counts = vectorizer.fit_transform(train_data)
test_counts = vectorizer.transform(test_data)
```

## 🤖 Model Training and Prediction

Train a **Multinomial Naive Bayes** classifier:

```python
from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()
classifier.fit(train_counts, train_labels)
predictions = classifier.predict(test_counts)
```

## ✅ Evaluation

Evaluate model accuracy:

```python
from sklearn.metrics import accuracy_score

print("Accuracy:", accuracy_score(test_labels, predictions))
```

## 📁 Project Files

- `Naive Bayes Classification.ipynb`: Notebook with all analysis steps.

## 📦 Technologies Used

- Python 3
- Jupyter Notebook
- pandas
- scikit-learn

## 🚀 How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/tweet-location-naive-bayes.git
cd tweet-location-naive-bayes
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Launch notebook:

```bash
jupyter notebook "Naive Bayes Classification.ipynb"
```

## 🔮 Future Improvements

- Use **TF-IDF** vectorization.
- Perform advanced tweet cleaning (remove emojis, URLs, stopwords).
- Test additional classifiers (e.g., Logistic Regression, SVM).
- Visualize distinctive word usage per city.
- Expand dataset to additional cities.

## 📌 Disclaimer

This is a simplified, educational project. Real-world applications would require more robust preprocessing and possibly larger datasets.
