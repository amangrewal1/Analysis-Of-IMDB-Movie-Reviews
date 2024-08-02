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

## I. Summary
The purpose of this project was to analyze the effect increasing the amount of training data has on testing accuracy. We chose to analyze this relationship by use of the IMDB Movie Review dataset. The classifiers that we analyzed were k Nearest Neighbors, Decision Tree, Feed Forward Neural Network, and Logistic Regression. Through this project, we were able to observe that the increase of testing accuracy in relation to the amount of training data added plateaus for some classifiers. Thus in some classifiers, the benefit of adding more training data becomes negligible once a certain threshold is passed.

## II. Data Description
Data Exploration
In our project, we used the IMDb movie review dataset. This dataset consists of 50,000 opinionated movie reviews evenly split between positive and negative reviews.
To get a better understanding of the dataset, we examined the most frequently used words in both positive and negative reviews.


