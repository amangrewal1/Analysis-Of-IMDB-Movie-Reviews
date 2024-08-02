from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def train_nn(X_train, X_test, y_train, y_test):
    clf = MLPClassifier(hidden_layer_sizes=(5,), activation='relu', solver='adam', alpha=0.0001, max_iter=200, random_state=42)
    clf.fit(X_train, y_train)
    train_accuracy = accuracy_score(y_train, clf.predict(X_train))
    test_accuracy = accuracy_score(y_test, clf.predict(X_test))
    print(f'Neural Network - Training accuracy: {train_accuracy:.2f}, Testing accuracy: {test_accuracy:.2f}')
    return clf
