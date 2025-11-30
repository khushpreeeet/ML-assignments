
# svm_solution.py
"""
SVM Lab Assignment Complete Solution
"""

from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np

#######################
# PART 1 – IRIS DATASET
#######################

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

kernels = ['linear', 'poly', 'rbf']
for k in kernels:
    if k == 'poly':
        model = SVC(kernel=k, degree=3)
    else:
        model = SVC(kernel=k)

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print("\n==== Kernel:", k, "====")
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Precision:", precision_score(y_test, pred, average='macro'))
    print("Recall:", recall_score(y_test, pred, average='macro'))
    print("F1:", f1_score(y_test, pred, average='macro'))
    print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

##############################
# PART 2 – BREAST CANCER DATA
##############################

data = load_breast_cancer()
X = data.data
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n=== Without Scaling ===")
svm_no = SVC(kernel='rbf')
svm_no.fit(X_train, y_train)
pred_no = svm_no.predict(X_test)
print("Train Acc:", accuracy_score(y_train, svm_no.predict(X_train)))
print("Test Acc:", accuracy_score(y_test, pred_no))

# With scaling
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

print("\n=== With Scaling ===")
svm_scaled = SVC(kernel='rbf')
svm_scaled.fit(X_train_scaled, y_train)
pred_scaled = svm_scaled.predict(X_test_scaled)
print("Train Acc:", accuracy_score(y_train, svm_scaled.predict(X_train_scaled)))
print("Test Acc:", accuracy_score(y_test, pred_scaled))
