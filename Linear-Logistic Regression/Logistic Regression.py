##Logistic Regression

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split

# Logistic Regression 

## Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


df = pd.read_csv("diabetes.csv")
df.head()

##  Model
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20, random_state=1)

log_model = LogisticRegression().fit(X_train, y_train)
log_model.intercept_
log_model.coef_

## Prediction
y_pred = log_model.predict(X_train)
y_pred[0:10]
y_train[0:10]

log_model.predict_proba(X_train)[0:10]

# Train Accuracy
y_pred = log_model.predict(X_train)
accuracy_score(y_train, y_pred)

# Test
# AUC Score için y_prob
y_prob = log_model.predict_proba(X_test)[:, 1]

# Diğer metrikler için y_pred
y_pred = log_model.predict(X_test)

# CONFUSION MATRIX
def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

plot_confusion_matrix(y_test, y_pred)

plot_confusion_matrix(y_test, y_pred)


# ACCURACY
accuracy_score(y_test, y_pred)

# PRECISION
precision_score(y_test, y_pred)

# RECALL
recall_score(y_test, y_pred)

# F1
f1_score(y_test, y_pred)

# ROC CURVE
plot_roc_curve(log_model, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()

# AUC
roc_auc_score(y_test, y_prob)

# Classification report
print(classification_report(y_test, y_pred))


# ROC AUC
y_prob = log_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)

## # Model Validation: Holdout

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)


log_model = LogisticRegression().fit(X_train, y_train)

y_pred = log_model.predict(X_test)
# AUC Score 
y_prob = log_model.predict_proba(X_test)[:, 1]
# Classification report
print(classification_report(y_test, y_pred))
# ROC Curve
plot_roc_curve(log_model, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()

roc_auc_score(y_test, y_prob)

# Model Validation: 10-Fold Cross Validation


y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

log_model = LogisticRegression().fit(X, y)

cv_results = cross_validate(log_model,X, y, cv=5, scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_precision'].mean()
cv_results['test_recall'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# Prediction for A New Observation

X.columns

random_user = X.sample(1, random_state=44)

log_model.predict(random_user)""