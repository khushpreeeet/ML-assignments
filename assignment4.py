
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge, Lasso, LinearRegression, RidgeCV, LassoCV, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.datasets import load_boston, load_iris

# Q1
np.random.seed(0)
X = np.random.rand(100, 7)
for i in range(1, 7):
    X[:, i] = X[:, i-1] + np.random.normal(0, 0.01, 100)
y = 5*X[:, 0] + 3*X[:, 1] + np.random.normal(0, 0.1, 100)

def ridge_regression_gd(X, y, lr, lam, epochs=1000):
    n, m = X.shape
    X_b = np.c_[np.ones((n, 1)), X]
    theta = np.random.randn(m+1, 1)
    y = y.reshape(-1, 1)
    for _ in range(epochs):
        gradients = (2/n) * X_b.T.dot(X_b.dot(theta) - y) + 2 * lam * theta
        theta -= lr * gradients
    return theta

best_r2 = -1
best_params = None
for lr in [0.0001, 0.001, 0.01, 0.1, 1, 10]:
    for lam in [1e-15, 1e-10, 1e-5, 1e-3, 0.1, 10, 20]:
        theta = ridge_regression_gd(X, y, lr, lam)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        y_pred = X_b.dot(theta)
        r2 = r2_score(y, y_pred)
        if r2 > best_r2:
            best_r2 = r2
            best_params = (lr, lam)

# Q2
url = "https://drive.google.com/uc?id=1qzCKF6JKKMB0p7u1ILy8tdmRk3vEbGvE"
df = pd.read_csv(url)
df = df.dropna()
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
X = df.drop('Salary', axis=1)
y = df['Salary']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
models = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(alpha=0.5748),
    'Lasso': Lasso(alpha=0.5748)
}
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = r2_score(y_test, y_pred)

# Q3
boston = load_boston()
X, y = boston.data, boston.target
ridge_cv = RidgeCV(alphas=[0.1, 1, 10], cv=5)
ridge_cv.fit(X, y)
lasso_cv = LassoCV(alphas=[0.1, 1, 10], cv=5)
lasso_cv.fit(X, y)

# Q4
iris = load_iris()
X, y = iris.data, iris.target
model = LogisticRegression(multi_class='ovr', max_iter=200)
model.fit(X, y)
y_pred = model.predict(X)
acc = accuracy_score(y, y_pred)
