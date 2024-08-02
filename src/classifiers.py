from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def logistic_classification(X_train, X_test, y_train, y_test):
    classifier = LogisticRegression(penalty='l1', fit_intercept=True, solver='liblinear')
    classifier.fit(X_train, y_train)
    train_accuracy = accuracy_score(y_train, classifier.predict(X_train))
    test_accuracy = accuracy_score(y_test, classifier.predict(X_test))
    print(f'Logistic Regression - Training accuracy: {train_accuracy:.2f}, Testing accuracy: {test_accuracy:.2f}')
    return classifier

def knn_classification(X_train, X_test, y_train, y_test):
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)
    train_accuracy = accuracy_score(y_train, classifier.predict(X_train))
    test_accuracy = accuracy_score(y_test, classifier.predict(X_test))
    print(f'k-NN - Training accuracy: {train_accuracy:.2f}, Testing accuracy: {test_accuracy:.2f}')
    return classifier

def decision_tree_classification(X_train, X_test, y_train, y_test):
    classifier = DecisionTreeClassifier(random_state=42)
    classifier.fit(X_train, y_train)
    train_accuracy = accuracy_score(y_train, classifier.predict(X_train))
    test_accuracy = accuracy_score(y_test, classifier.predict(X_test))
    print(f'Decision Tree - Training accuracy: {train_accuracy:.2f}, Testing accuracy: {test_accuracy:.2f}')
    return classifier
