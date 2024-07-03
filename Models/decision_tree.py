
import warnings
import joblib
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from skompiler import skompile


pd.set_option('display.max_columns', None)
warnings.simplefilter(action='ignore', category=Warning)


# Modeling using CART
df = pd.read_csv("dsmlbc6/datasets/hafta8/diabetes.csv")
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

## Model
cart_model = DecisionTreeClassifier(random_state=17).fit(X, y)

# y_pred for Confusion matrix 
y_pred = cart_model.predict(X)
y_pred  = cart_model.predict(X)
print(y_pred)


# AUC için y_prob:
y_prob = cart_model.predict_proba(X)[:,1]
print(y_prob)

# Confusion matrix
print(classification_report(y, y_pred))

# AUC
print(roc_auc_score(y, y_prob))


# Validation with holdout
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
cart_model = DecisionTreeClassifier(random_state=17).fit(X_train,y_train)

## Train Error
y_pred = cart_model.predict(X_train)
y_prob = cart_model.predict_proba(X_train)[:,1]
print(classification_report(y_train, y_pred))
print(roc_auc_score(y_train, y_prob))

## Test Error
y_pred = cart_model.predict(X_test)
y_prob = cart_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test, y_prob))

# Validation with CC
cart_model = DecisionTreeClassifier(random_state=17).fit(X, y)
cv_results = cross_validate(cart_model, X, y, cv=10,scoring=["accuracy", "f1", "roc_auc"])

print(cv_results['test_accuracy'].mean())
print(cv_results['test_f1'].mean())
print(cv_results['test_roc_auc'].mean())


# Hyperparameter Optimization with GridSearchCV
print(cart_model.get_params())

# Hyperparameters 
cart_params = {'max_depth': range(1, 11), "min_samples_split": range(2, 20)}

# GridSearchCV
cart_best_grid = GridSearchCV(cart_model,  cart_params,cv=5, n_jobs=-1,verbose=True).fit(X, y)

print(cart_best_grid.best_params_)
print(cart_best_grid.best_score_)
random = X.sample(1, random_state=45)
print(cart_best_grid.predict(random))

# Final Model
cart_final = DecisionTreeClassifier(**cart_best_grid.best_params_, random_state=17).fit(X, y)
print(cart_final.get_params())
cart_final = cart_model.set_params(**cart_best_grid.best_params_).fit(X, y)

# CV Error
cv_results = cross_validate(cart_final, X, y,cv=10,scoring=["accuracy", "f1", "roc_auc"])

print(cv_results['test_accuracy'].mean())
print(cv_results['test_f1'].mean())
print(cv_results['test_roc_auc'].mean())

# Feature Importance
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(cart_final, X, 15)


# Analyzing Model Complexity with Learning Curves 
train_score, test_score = validation_curve( cart_final, X=X, y=y, param_name='max_depth',
    param_range=range(1, 11),scoring="roc_auc", cv=10)

mean_train_score = np.mean(train_score, axis=1)
mean_test_score = np.mean(test_score, axis=1)

plt.plot(range(1, 11), mean_train_score,
         label="Training Score", color='b')

plt.plot(range(1, 11), mean_test_score,
         label="Validation Score", color='g')

plt.title("Validation Curve for CART")
plt.xlabel("Number of max_depth")
plt.ylabel("AUC")
plt.tight_layout()
plt.legend(loc='best')
plt.show()

#Visualizing the Decision Tree
def tree_graph(model, col_names, file_name):
    tree_str = export_graphviz(model, feature_names=col_names, filled=True, out_file=None)
    graph = pydotplus.graph_from_dot_data(tree_str)
    graph.write_png(file_name)

tree_graph(model=cart_final, col_names=X.columns, file_name="cart_final.png")

#Extracting Decision Rules
tree_rules = export_text(cart_model, feature_names=list(X.columns))
print(tree_rules)

# Saving and Loading Model
joblib.dump(cart_final, "cart_final.pkl")
cart_model_from_disk = joblib.load("cart_final.pkl")

x = [12, 13, 20, 23, 4, 55, 12, 7]
cart_model_from_disk.predict(pd.DataFrame(x).T)
