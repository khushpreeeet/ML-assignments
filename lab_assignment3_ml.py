
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

# ------------------ Q1: K-Fold Cross Validation ------------------
def q1_house_price_kfold(data_path):
    df = pd.read_csv(data_path)
    X = df.drop("price", axis=1).values
    y = df["price"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_r2 = -np.inf
    best_beta = None

    for train_index, test_index in kf.split(X_scaled):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y[train_index], y[test_index]

        beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train
        y_pred = X_test @ beta
        r2 = r2_score(y_test, y_pred)

        print("R2 Score:", r2)
        if r2 > best_r2:
            best_r2 = r2
            best_beta = beta

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    y_pred = X_test @ best_beta
    print("Final R2 Score (best beta on 70/30 split):", r2_score(y_test, y_pred))

# ------------------ Q2: Gradient Descent with Validation ------------------
def gradient_descent(X, y, lr, iterations=1000):
    m, n = X.shape
    beta = np.zeros(n)
    for _ in range(iterations):
        gradient = -(2/m) * X.T @ (y - X @ beta)
        beta -= lr * gradient
    return beta

def q2_house_price_gd(data_path):
    df = pd.read_csv(data_path)
    X = df.drop("price", axis=1).values
    y = df["price"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train_val, X_test, y_train_val, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

    lrs = [0.001, 0.01, 0.1, 1]
    best_r2 = -np.inf
    best_beta = None

    for lr in lrs:
        beta = gradient_descent(X_train, y_train, lr)
        r2_val = r2_score(y_val, X_val @ beta)
        r2_test = r2_score(y_test, X_test @ beta)
        print(f"LR: {lr}, R2 Validation: {r2_val}, R2 Test: {r2_test}")

        if r2_val > best_r2:
            best_r2 = r2_val
            best_beta = beta

    print("Best Beta coefficients:", best_beta)

# ------------------ Q3: Preprocessing & PCA ------------------
def q3_car_price(data_path):
    col_names = ["symboling", "normalized_losses", "make", "fuel_type", "aspiration",
                 "num_doors", "body_style", "drive_wheels", "engine_location", "wheel_base",
                 "length", "width", "height", "curb_weight", "engine_type", "num_cylinders",
                 "engine_size", "fuel_system", "bore", "stroke", "compression_ratio",
                 "horsepower", "peak_rpm", "city_mpg", "highway_mpg", "price"]
    
    df = pd.read_csv(data_path, names=col_names, na_values="?")
    df = df.dropna(subset=["price"])
    df.fillna(df.median(numeric_only=True), inplace=True)

    num_map = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
               "eight": 8, "twelve": 12}
    df["num_doors"] = df["num_doors"].map(num_map).fillna(df["num_doors"])
    df["num_cylinders"] = df["num_cylinders"].map(num_map).fillna(df["num_cylinders"])

    df = pd.get_dummies(df, columns=["body_style", "drive_wheels"])

    for col in ["make", "aspiration", "engine_location", "fuel_type"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    df["fuel_system"] = df["fuel_system"].apply(lambda x: 1 if "pfi" in str(x).lower() else 0)
    df["engine_type"] = df["engine_type"].apply(lambda x: 1 if "ohc" in str(x).lower() else 0)

    X = df.drop("price", axis=1).apply(pd.to_numeric, errors="coerce").fillna(0)
    y = pd.to_numeric(df["price"], errors="coerce")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    print("R2 Score without PCA:", r2_score(y_test, lr.predict(X_test)))

    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X_scaled)
    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_pca, y, test_size=0.3, random_state=42)
    lr.fit(X_train_p, y_train_p)
    print("R2 Score with PCA:", r2_score(y_test_p, lr.predict(X_test_p)))

if __name__ == "__main__":
    
    print("Call q1_house_price_kfold('house_price.csv'), q2_house_price_gd('house_price.csv'), q3_car_price('imports-85.data') to run each solution.")
