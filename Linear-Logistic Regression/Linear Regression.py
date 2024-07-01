import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score

pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Linear Regression Example

## Error function
def mse(actual, predicted):
    return np.mean((actual - predicted) ** 2)


def rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))


def mae(actual, predicted):
    return np.mean(np.abs(actual - predicted))


## Simple Linear Regr.ession with OLS Using Scikit-Learn
df = pd.read_csv("advertising.csv")
df.head(10)

X = df[["TV"]]
y = df[["sales"]]

## Model
reg_model = LinearRegression().fit(X, y)

##bias
reg_model.intercept_[0]
## weight
reg_model.coef_[0][0]


## Prediction
reg_model.intercept_[0] + reg_model.coef_[0][0] * 150
reg_model.intercept_[0] + reg_model.coef_[0][0] * 500

## Develop Model
g = sns.regplot(x=X, y=y, scatter_kws={'color': 'b', 's': 9},
                ci=False, color="r")
g.set_title(f"Model Denklemi: Sales = {round(reg_model.intercept_[0], 2)} + TV*{round(reg_model.coef_[0][0], 2)}")
g.set_ylabel("Satış Sayısı")
g.set_xlabel("TV Harcamaları")
plt.xlim(-10, 310)
plt.ylim(bottom=0)
plt.show()

## Error

### MSE
y_pred = reg_model.predict(X)
mean_squared_error(y, y_pred)
### RMSE
np.sqrt(mean_squared_error(y, y_pred))
### MAEmean_absolute_error(y, y_pred)

### R-SQUARE
reg_model.score(X, y)

# Multiple Linear Regression eXAMPLE
df = pd.read_csv("advertising.csv")
X = df.drop('sales', axis=1)
y = df[["sales"]]

## Model
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=40)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

## bias
reg_model.intercept_

## coefficients (wweights)
reg_model.coef_

## Prediction

## Train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))

## TRAIN RKARE
reg_model.score(X_train, y_train)

## Test RMSE
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

## Test R-SQUARE
reg_model.score(X_test, y_test)

## Cross Validation RMSE
np.mean(np.sqrt(-cross_val_score(reg_model, X, y, cv=10, scoring="neg_mean_squared_error")))

# Simple Linear Regression with Gradient Descent from Scratch

## Cost function
def cost_function(Y, b, w, X):
    m = len(Y)  # gözlem sayısı
    sse = 0  # toplam hata
    # butun gozlem birimlerini gez:
    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        sse += (y_hat - y) ** 2
    mse = sse / m
    return mse

## Update result of gradient descent
def update_weights(Y, b, w, X, learning_rate):
    m = len(Y)
    b_deriv_sum = 0
    w_deriv_sum = 0
    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        b_deriv_sum += (y_hat - y)
        w_deriv_sum += (y_hat - y) * X[i]
    new_b = b - (learning_rate * 1 / m * b_deriv_sum)
    new_w = w - (learning_rate * 1 / m * w_deriv_sum)
    return new_b, new_w

## Train function
def train(Y, initial_b, initial_w, X, learning_rate, num_iters):
    print("Starting gradient descent at b = {0}, w = {1}, mse = {2}".format(initial_b, initial_w, cost_function(Y, initial_b, initial_w, X)))
    b = initial_b
    w = initial_w
    cost_history = []

    for i in range(num_iters):
        b, w = update_weights(Y, b, w, X, learning_rate)
        mse = cost_function(Y, b, w, X)
        cost_history.append(mse)


        if i % 100 == 0:
            print("iter={:d}    b={:.2f}    w={:.4f}    mse={:.4}".format(i, b, w, mse))

    print("After {0} iterations b = {1}, w = {2}, mse = {3}".format(num_iters, b, w, cost_function(Y, b, w, X)))
    return cost_history, b, w


X = df["radio"]
Y = df["sales"]

# hyperparameters
learning_rate = 0.001
initial_b = 0.001
initial_w = 0.001
num_iters = 10000

train(Y, initial_b, initial_w, X, learning_rate, num_iters)