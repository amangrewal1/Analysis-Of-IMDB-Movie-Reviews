from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def ensemble_method(X_train, X_test, y_train, y_test):
    clf1 = LogisticRegression(penalty='l1', fit_intercept=True, solver='liblinear')
    clf2 = DecisionTreeClassifier(random_state=42)
    clf3 = KNeighborsClassifier(n_neighbors=5)

    ensemble_clf = VotingClassifier(estimators=[
        ('lr', clf1), ('dt', clf2), ('knn', clf3)], voting='hard')

    ensemble_clf.fit(X_train, y_train)
    train_accuracy = accuracy_score(y_train, ensemble_clf.predict(X_train))
    test_accuracy = accuracy_score(y_test, ensemble_clf.predict(X_test))
    print(f'Ensemble - Training accuracy: {train_accuracy:.2f}, Testing accuracy: {test_accuracy:.2f}')
    return ensemble_clf
