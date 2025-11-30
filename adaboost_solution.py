
# adaboost_solution.py
"""
Complete AdaBoost solution for SMS Spam Classification
(Assignment Reference)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import AdaBoostClassifier

# ---- Load dataset ----
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[['v1','v2']]
df.columns = ['label','text']
df['label'] = df['label'].map({'ham':0,'spam':1})

# ---- Preprocessing ----
X = df['text']
y = df['label']

tf = TfidfVectorizer(stop_words='english')
X_vec = tf.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X_vec,y,test_size=0.2,random_state=42)

# ---- Decision stump ----
stump = DecisionTreeClassifier(max_depth=1)
stump.fit(X_train,y_train)

print("Baseline Train:",accuracy_score(y_train,stump.predict(X_train)))
print("Baseline Test :",accuracy_score(y_test,stump.predict(X_test)))
print("Confusion:\n",confusion_matrix(y_test,stump.predict(X_test)))

# ---- Manual AdaBoost ----
n_rounds = 15
N = X_train.shape[0]
w = np.ones(N)/N
alphas=[]
errors=[]

for t in range(n_rounds):
    stump = DecisionTreeClassifier(max_depth=1)
    stump.fit(X_train,y_train,sample_weight=w)
    pred = stump.predict(X_train)
    err = np.sum(w*(pred!=y_train))
    alpha = 0.5*np.log((1-err)/err)
    w = w*np.exp(-alpha*y_train*(2*pred-1))
    w = w/np.sum(w)
    errors.append(err)
    alphas.append(alpha)
    print(f"Iter {t+1}: error={err:.4f}, alpha={alpha:.4f}")

plt.figure()
plt.plot(errors)
plt.title("Error vs Round")
plt.savefig("error.png")

plt.figure()
plt.plot(alphas)
plt.title("Alpha vs Round")
plt.savefig("alpha.png")

# ---- Final AdaBoost (sklearn) ----
ada = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100, learning_rate=0.6)
ada.fit(X_train,y_train)

print("AdaBoost Train:",accuracy_score(y_train,ada.predict(X_train)))
print("AdaBoost Test :",accuracy_score(y_test,ada.predict(X_test)))
print("Ada Confusion:\n",confusion_matrix(y_test,ada.predict(X_test)))
