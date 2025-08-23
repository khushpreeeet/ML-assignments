# bike_buyers_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, KBinsDiscretizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jaccard


df = pd.read_csv("AWCustomers.csv")

# ======================
# Part I: Feature Selection & Cleaning

# Convert BirthDate -> Age
df["BirthDate"] = pd.to_datetime(df["BirthDate"], errors="coerce")
df["Age"] = (pd.to_datetime("today") - df["BirthDate"]).dt.days // 365

# Select useful attributes
selected_columns = [
    "Gender", "MaritalStatus", "HomeOwnerFlag",
    "NumberCarsOwned", "NumberChildrenAtHome", "TotalChildren",
    "Education", "Occupation", "YearlyIncome", "Age"
]
df_selected = df[selected_columns].copy()


# Part II: Preprocessing & Transformation

# (a) Handle Null values
for col in df_selected.select_dtypes(include=[np.number]).columns:
    df_selected[col].fillna(df_selected[col].median(), inplace=True)
for col in df_selected.select_dtypes(include=["object"]).columns:
    df_selected[col].fillna(df_selected[col].mode()[0], inplace=True)

# (b) Normalization (0-1 scaling)
numeric_cols = ["NumberCarsOwned", "NumberChildrenAtHome", "TotalChildren", "YearlyIncome", "Age"]
scaler = MinMaxScaler()
df_selected[numeric_cols] = scaler.fit_transform(df_selected[numeric_cols])

# (c) Discretization (binning)
age_bins = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="quantile")
df_selected["Age_binned"] = age_bins.fit_transform(df_selected[["Age"]])

income_bins = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="quantile")
df_selected["Income_binned"] = income_bins.fit_transform(df_selected[["YearlyIncome"]])

# (d) Standardization (Z-score)
std_scaler = StandardScaler()
df_selected[[col + "_std" for col in numeric_cols]] = std_scaler.fit_transform(df_selected[numeric_cols])

# (e) One-Hot Encoding
categorical_cols = ["Gender", "MaritalStatus", "HomeOwnerFlag", "Education", "Occupation"]
ohe = OneHotEncoder(sparse=False, drop="first")
ohe_df = pd.DataFrame(ohe.fit_transform(df_selected[categorical_cols]),
                      columns=ohe.get_feature_names_out(categorical_cols))
df_transformed = pd.concat([df_selected.reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1)


# Part III: Proximity & Correlation


# (a) Similarity between two customers
# Example: Compare first two rows after transformation
obj1 = df_transformed.iloc[0].values.reshape(1, -1)
obj2 = df_transformed.iloc[1].values.reshape(1, -1)

# Simple Matching Coefficient (SMC) for binary/nominal
def simple_matching(x, y):
    return np.sum(x == y) / len(x)

smc = simple_matching(df_transformed.iloc[0].values, df_transformed.iloc[1].values)

# Jaccard Similarity (for binary attributes only)
# Convert categorical OHE part into binary subset
binary_part = ohe_df.values
jaccard_sim = 1 - jaccard(binary_part[0], binary_part[1])

# Cosine Similarity
cos_sim = cosine_similarity(obj1, obj2)[0][0]

# (b) Correlation between Commute Distance and Yearly Income

corr = df_transformed["YearlyIncome"].corr(df_transformed["Age"])


print("Similarity & Correlation Results:")
print(f"Simple Matching Coefficient (Obj1 vs Obj2): {smc:.4f}")
print(f"Jaccard Similarity (Obj1 vs Obj2): {jaccard_sim:.4f}")
print(f"Cosine Similarity (Obj1 vs Obj2): {cos_sim:.4f}")
print(f"Correlation (YearlyIncome vs Age): {corr:.4f}")


df_transformed.to_csv("AWCustomers_transformed.csv", index=False)