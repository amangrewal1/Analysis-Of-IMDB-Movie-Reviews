import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

def create_bow_from_reviews(dataset):
    df = dataset.to_pandas()
    text = df['text'].tolist()
    y = df['label'].tolist()

    vectorizer = CountVectorizer(stop_words='english', min_df=0.02, ngram_range=(1, 1))
    X = vectorizer.fit_transform(text)
    return X, y, vectorizer

def preprocess_data(dataset):
    X, y, vectorizer = create_bow_from_reviews(dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, vectorizer
