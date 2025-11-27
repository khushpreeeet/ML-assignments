
##############################################
# Assignment - Naive Bayes & GridSearchCV KNN
# Single .py File
##############################################

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import warnings
import math
warnings.filterwarnings("ignore")

##############################################
# (1) Gaussian Naive Bayes (Manual + In-built)
##############################################

iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.priors = {}

        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = X_c.mean(axis=0)
            self.var[c] = X_c.var(axis=0)
            self.priors[c] = X_c.shape[0] / X.shape[0]

    def gaussian_probability(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def predict(self, X):
        y_pred = []
        for x in X:
            posteriors = []

            for c in self.classes:
                prior = np.log(self.priors[c])
                class_conditional = np.sum(np.log(self.gaussian_probability(c, x)))
                posterior = prior + class_conditional
                posteriors.append(posterior)

            y_pred.append(self.classes[np.argmax(posteriors)])
        return np.array(y_pred)

gnb_manual = GaussianNaiveBayes()
gnb_manual.fit(X_train, y_train)
y_pred_manual = gnb_manual.predict(X_test)

print("\n(i) Step-by-step Gaussian Naive Bayes")
print("Accuracy:", accuracy_score(y_test, y_pred_manual))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_manual))
print("Classification Report:\n", classification_report(y_test, y_pred_manual, target_names=target_names))

gnb_builtin = GaussianNB()
gnb_builtin.fit(X_train, y_train)
y_pred_builtin = gnb_builtin.predict(X_test)

print("\n(ii) In-built GaussianNB")
print("Accuracy:", accuracy_score(y_test, y_pred_builtin))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_builtin))
print("Classification Report:\n", classification_report(y_test, y_pred_builtin, target_names=target_names))


##############################################
# (2) GRIDSEARCHCV for KNN
##############################################

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier()

param_grid = {
    'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15],
    'weights': ['uniform', 'distance']
}

grid_search = GridSearchCV(
    estimator=knn,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)

print("\nBest Parameters Found:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test_scaled)

print("\nTest Set Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

results = pd.DataFrame(grid_search.cv_results_)
print("\nAll Grid Search Results (Top 5):")
print(results[['param_n_neighbors', 'param_weights', 'mean_test_score']].head())
