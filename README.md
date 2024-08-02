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

#### Previous Research
The IMDb dataset is commonly used in machine learning studies. One such study is "Fast and accurate sentiment classification using an enhanced Naive Bayes model" by Vivek Narayanan, Ishan Arora and Arjun Bhatia. The authors used this dataset to train a Naive Bayes classifier with some modifications and achieved an accuracy of 88.8%.

### III. Classifiers
The following classifiers were analyzed in this project:

#### K Nearest Neighbors Classifier
The K-Nearest Neighbors (kNN) classifier is a supervised machine learning algorithm used for both classification and regression tasks. It does not make any assumptions about the data itself and does not build a model during the training phase. kNN rather stores all data points and utilizes them during the prediction phase.
The classifier works by looking at the nearest k neighbors and then classifying the current data point as the majority class label of those k neighbors. k is also known as the hyper parameter of this classifier since it is a parameter for the classifier that can vary according to various factors such as dataset size, data type, etc.

#### Logistic Regression Classifier
The Logistic Regression Classifier is oftentimes used for binary or categorical classification problems. At the base, the classifier transforms the input into a probability and decides on a certain threshold. This threshold is used to determine the classification of the label.

#### Decision Tree Classifier
The Decision Tree Classifier works by building a flowchart-like tree where every node splits a feature on a certain threshold. This threshold is determined by finding the best gini score from all possible thresholds.
Once the tree is built based on the training data, it can then be used to predict the class label of any testing data point by starting from the root node, and falling down the branches of the tree.

#### Feed Forward Neural Network Classifier
A Neural Network has multiple layers of interconnected nodes. The first layer is the input layer, which is followed by one or more hidden layers that finally lead to an output layer.
The input layer’s nodes take in the input and calculate an output to the node it is connected to in the next layer based on some biases and weights. This process continues until reaching the output layer which then makes a prediction on the class label of the input. The biases and weights of the intermediate layers are determined by the training phase, where each of them are tuned.

#### Ensemble Methods
Ensemble methods combine multiple classifiers to improve the accuracy and robustness of predictions. In this project, we used a Voting Classifier that combines Logistic Regression, Decision Tree, and k-Nearest Neighbors classifiers. The combined prediction is made based on the majority vote of these individual classifiers.

#### GPU Acceleration
To expedite the training and evaluation process, we utilized GPU acceleration. Specifically, we leveraged TensorFlow to train a Feed Forward Neural Network on a GPU, which significantly reduced the computation time compared to CPU-based training.

### IV. Experimental Setup
The project was conducted in Google Colab, using the Interactive Python Notebook. To start, pip was used to install the datasets library (!pip install datasets), which gives access to the Hugging Face Data Sets. The imdb dataset was imported using the function “load_dataset”.
Our goal was to generate and analyze the learning curves for all the classifiers listed above. A learning curve is a graph of Accuracy versus dataset size. To generate a learning curve for a certain classifier, we calculate the training and testing accuracy for that classifier at a certain dataset size. Additionally, we randomly sample for training and testing data at each dataset size step. Fortunately, this is all done for us using the learning_curve function in sklearn.

For each classifier, the sklearn learning_curve function tracked the learning and validation accuracy while increasing the amount of training data used. For the K Nearest Neighbors Classifier, the hyperparameter, k, was chosen to be 30. For the Feed Forward Neural Network Classifier, the hyperparameter is the amount of hidden layers and was chosen to be three. The hyperparameter for the Decision Tree Classifier is max depth and was chosen to be five. The hyperparameters for the Logistical Regression Classifier are solver, fit_intercept, and penalty which were chosen to be liblinear, true, and 11 respectively.
The parameters passed to the learning_curve function were learning_curve(classifier, X, y, train_sizes=np.linspace(10000, 40000, 5, dtype=int), cv=5)
where classifier is the instance of the classifier we are testing.

To leverage GPU acceleration, we used TensorFlow for training the Feed Forward Neural Network. This significantly reduced the computation time and allowed us to handle larger datasets more efficiently.

Finally, a separate plotting function, LearningCurveDisplay, was also imported from sklearn to display the learning curve graphically with the help of matplotlib.

### V. Experimental Results

#### K Nearest Neighbors Classifier
To tune the hyper parameter, k, we first plotted k vs the accuracy of the model. The results can be seen in Figure 1. Based on the Figure 1, we decided to use k as 30 because of the diminishing returns in accuracy as we increased k further. The kNN classifier generally had a low average testing accuracy. kNN models can have lower testing accuracy compared to other models because they make decisions based on a small number of nearest neighbors. If the dataset is noisy or the classes are not well separated, the decision of kNN can be easily influenced by noise, leading to lower accuracy. Increasing the number of training samples used continued to have a positive effect on the testing accuracy. This is because as the number of training samples increases, the 30 closest neighbors are now closer and more specialized to the sample being predicted.
<img width="450" alt="Screenshot 2024-08-02 at 2 34 40 PM" src="https://github.com/user-attachments/assets/c64977ce-c2cb-4e19-a851-828135f966f9">

#### Logistical Regression
For the Logistical Regression Classifier, we noticed that the testing accuracy had a steady increase from datasets sized 10000 through 32500 and then increased at a slightly higher rate afterwards. In terms of training accuracy, we noticed a steady increase across the entire range of dataset sizes. This could indicate that the model is less likely to overfit data with larger datasets as the training data becomes more generalized with increasing dataset sizes.
<img width="460" alt="Screenshot 2024-08-02 at 2 35 19 PM" src="https://github.com/user-attachments/assets/735168cc-4653-4297-af54-0fb5286a6cca">

#### Decision Tree Classifier
The results for the Decision Tree Classifier surprised us as the testing accuracy started to have a rate of decrease in testing accuracy for medium sized datasets, but was in tune with the Logistical Regression Classifier and kNN for large sized datasets. Decision Trees can easily overfit to the training data, especially when the tree is allowed to grow without restrictions. For medium-sized datasets, if the decision tree is too complex (deep), it can start overfitting, leading to a decrease in testing accuracy. This is likely why you observed a dip in accuracy for medium-sized datasets. The overall accuracy was generally lower since Decision Trees can also be easily influenced by noise.
<img width="471" alt="Screenshot 2024-08-02 at 2 35 33 PM" src="https://github.com/user-attachments/assets/2f07f8e9-a1c8-45f0-936a-52d8140694f2">

#### Feed Forward Neural Network Classifier

For the Feed Forward Neural Network, the rate of increase starts to plateau for medium and large datasets. As the dataset size increases, the model has more examples to learn from, reducing the chance of overfitting. However, this can also lead to a plateau in testing accuracy as the model begins to generalize better and overfit less. Generally, the testing accuracy was high for this model.

<img width="463" alt="Screenshot 2024-08-02 at 2 36 35 PM" src="https://github.com/user-attachments/assets/7fcd7811-ad39-43c8-8335-d9b3dd04a074">

#### Ensemble Methods
The ensemble method combining Logistic Regression, Decision Tree, and k-Nearest Neighbors showed an improvement in testing accuracy. This is because the ensemble method leverages the strengths of each individual classifier, leading to a more robust and accurate prediction. The combined model tends to perform better on average than any single model.

#### GPU Acceleration
The use of GPU acceleration for training the Feed Forward Neural Network significantly reduced the training time. This allowed us to experiment with larger datasets and more complex models without a substantial increase in computation time. The speed-up was particularly noticeable during the training phase, where matrix operations are heavily utilized.

### VI. Conclusion
In conclusion, we observed that increasing the amount of training data generally improves testing accuracy, but the rate of improvement plateaus for some classifiers. The ensemble method provided a robust solution by combining multiple classifiers, and GPU acceleration was effective in reducing training time for complex models. Future work could involve exploring additional ensemble methods and further optimizing hyperparameters for each classifier.
