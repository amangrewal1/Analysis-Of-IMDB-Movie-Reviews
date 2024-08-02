# IMDB Classification Project

This project conducts an analysis of various classifiers on the IMDB Movie Review dataset, including Logistic Regression, k-Nearest Neighbors, Decision Trees, and Feed Forward Neural Networks. Ensemble methods are used to combine predictions, and GPU acceleration is utilized for faster model training.

## Project Structure

- `data/`: Contains the script to download and preprocess the dataset.
- `src/`: Contains the implementation of data preprocessing, classifiers, ensemble methods, training, and plotting functions.
- `main.py`: Main script to run the entire analysis.

## How to Run

1. Install the required packages: pip install -r requirements.txt
2. Run the main script:


## Report:

### I. Summary
The purpose of this project was to analyze the effect increasing the amount of training data has on testing accuracy. We chose to analyze this relationship by use of the IMDB Movie Review dataset. The classifiers that we analyzed were k Nearest Neighbors, Decision Tree, Feed Forward Neural Network, and Logistic Regression. Through this project, we were able to observe that the increase of testing accuracy in relation to the amount of training data added plateaus for some classifiers. Thus in some classifiers, the benefit of adding more training data becomes negligible once a certain threshold is passed.

### II. Data Description
Data Exploration
In our project, we used the IMDb movie review dataset. This dataset consists of 50,000 opinionated movie reviews evenly split between positive and negative reviews.
To get a better understanding of the dataset, we examined the most frequently used words in both positive and negative reviews.

<img width="949" alt="Screenshot 2024-08-02 at 2 29 49 PM" src="https://github.com/user-attachments/assets/ef374dbf-36f9-47af-94c3-0c093f7f9e7d">


This figure did not provide much insight into the differences between positive and negative reviews. Next, we examined the words that were most frequently used in positive reviews but not negative reviews and vice versa.

<img width="900" alt="Screenshot 2024-08-02 at 2 30 25 PM" src="https://github.com/user-attachments/assets/985deee3-53c3-48cb-8bd2-f952d9d77413">

From these figures, we can make observations such as words like “great” and “love” being indicative of positive reviews.
Additional Note: we observed that “br”, which represents the html <br> tag, is a part of our bag of words. We decided to not remove this tag as it represents additional data which may determine whether a review is negative or positive.

# Previous Research
The IMDb dataset is commonly used in machine learning studies. One such study is "Fast and accurate sentiment classification using an enhanced Naive Bayes model" by Vivek Narayanan, Ishan Arora and Arjun Bhatia. The authors used this dataset to train a Naive Bayes classifier with some modifications and achieved an accuracy of 88.8%.
# 






