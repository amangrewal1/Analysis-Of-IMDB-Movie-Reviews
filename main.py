from data.download_data import download_imdb_dataset
from src.data_preprocessing import preprocess_data
from src.classifiers import logistic_classification, knn_classification, decision_tree_classification
from src.ensemble import ensemble_method
from src.train import train_nn
from src.plot import plot_learning_curve
from sklearn.neural_network import MLPClassifier

def main():
    # Download and preprocess dataset
    dataset = download_imdb_dataset()
    X_train, X_test, y_train, y_test, _ = preprocess_data(dataset['train'])

    # Train classifiers
    logistic_classification(X_train, X_test, y_train, y_test)
    knn_classification(X_train, X_test, y_train, y_test)
    decision_tree_classification(X_train, X_test, y_train, y_test)

    # Train ensemble method
    ensemble_method(X_train, X_test, y_train, y_test)

    # Train neural network and plot learning curve
    nn_clf = train_nn(X_train, X_test, y_train, y_test)
    plot_learning_curve(nn_clf, X_train, y_train, "Neural Network Learning Curve")

if __name__ == "__main__":
    main()
