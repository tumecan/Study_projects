
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


pd.set_option('display.max_columns', None)
warnings.simplefilter(action='ignore', category=Warning)

df = pd.read_csv("dsmlbc6/datasets/hafta8/diabetes.csv")
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

# Random Forests

rf_model = RandomForestClassifier(random_state=17)
rf_params = {"max_depth": [5, 8, None], ## ne kadar aşağı inecek.
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20], ## yaprak sayısı
             "n_estimators": [100, 200, 500]} ## ağaç sayısı

rf_best_grid = GridSearchCV(rf_model, rf_params, cv = 5,n_jobs=-1,verbose=True).fit(X, y)
print(rf_best_grid.best_params_)
rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state= 17).fit(X,y)
cv_results = cross_validate(rf_final, X,y,cv = 5,scoring=["accuracy", "f1", "roc_auc"])

print(cv_results['test_accuracy'].mean())
print(cv_results['test_f1'].mean())
print(cv_results['test_roc_auc'].mean())

# GBM Model
gbm_model = GradientBoostingClassifier(random_state=17)
gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8],
              "n_estimators": [500, 1000],
              "subsample": [1, 0.5, 0.7]}

gbm_best_grid = GridSearchCV(gbm_model,gbm_params, cv=5,n_jobs=-1,verbose=True).fit(X, y)
gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_,random_state=17, ).fit(X, y)
cv_results = cross_validate(gbm_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])

print(cv_results['test_accuracy'].mean())
print(cv_results['test_f1'].mean())
print(cv_results['test_roc_auc'].mean())

# XGBoost
xgboost_model = XGBClassifier(random_state=17)
xgboost_params = {"learning_rate": [0.1, 0.01, 0.001],
                  "max_depth": [5, 8, 12, 15, 20],
                  "n_estimators": [100, 500, 1000],
                  "colsample_bytree": [0.5, 0.7, 1]}

xgboost_best_grid = GridSearchCV(xgboost_model,xgboost_params,cv=5, n_jobs=-1, verbose=True).fit(X, y)
print(xgboost_best_grid.best_score_)
xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X, y)
cv_results = cross_validate(xgboost_final,X, y,cv=10,scoring=["accuracy", "f1", "roc_auc"])

print(cv_results['test_accuracy'].mean())
print(cv_results['test_f1'].mean())
print(cv_results['test_roc_auc'].mean())


# LightGBM
lgbm_model = LGBMClassifier(random_state=17)
lgbm_params = {"learning_rate": [0.001, 0.01, 0.02, 0.03, 0.05, 0.1],
        "n_estimators": [100, 250, 350, 500, 750, 1000],
        "colsample_bytree": [0.5, 0.8, 0.7, 0.6, 1],
        "min_child_samples": [20, 50, 100],  # min_data_in_leaf
        "min_split_gain": [0.0, 0.1, 0.2],  # min_gain_to_split
        "num_leaves": [31, 50, 100]  # num_leaves}
}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv = 10, n_jobs=-1, verbose=True).fit(X, y)
lgbm_final= lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state = 42).fit(X,y)
cv_resutls= cross_validate(lgbm_final, X,y, cv = 10, scoring=["accuracy", "f1", "roc_auc"])

print(cv_results['test_accuracy'].mean())
print(cv_results['test_f1'].mean())
print(cv_results['test_roc_auc'].mean())


# CatBoost
catboost_model = CatBoostClassifier(random_state=17, verbose=False)
catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}

catboost_best_grid = GridSearchCV(catboost_model, catboost_params,cv=5,n_jobs=-1,verbose=True).fit(X, y)
catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_,random_state=17).fit(X, y)
cv_results = cross_validate(catboost_final, X, y, cv=10,scoring=["accuracy", "f1", "roc_auc"])


print(cv_results['test_accuracy'].mean())
print(cv_results['test_f1'].mean())
print(cv_results['test_roc_auc'].mean())

# Feature Importance
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_final, X)
plot_importance(gbm_final, X)
plot_importance(xgboost_final, X)
plot_importance(lgbm_final, X)
plot_importance(catboost_final, X)

# Hyperparameter Optimization with RandomSearchCV 
rf_model = RandomForestClassifier(random_state=17)
rf_random_params = {"max_depth": np.random.randint(5, 50, 10),
                    "max_features": [3, 5, 7, "auto", "sqrt"],
                    "min_samples_split": np.random.randint(2, 50, 20),
                    "n_estimators": [int(x) for x in np.linspace(start=200, stop=1500, num=10)]}

rf_random = RandomizedSearchCV(estimator=rf_model,
                               param_distributions=rf_random_params,
                               n_iter=100,  # denenecek parametre sayısı
                               cv=10,
                               verbose=True,
                               random_state=42,
                               n_jobs=-1)

rf_random.fit(X, y)
print(rf_random.best_params_)
print(rf_random.best_score_)
rf_random_final = rf_model.set_params(**rf_random.best_params_,random_state=17).fit(X, y)
cv_results = cross_validate(rf_random_final,X, y,cv=5,scoring=["accuracy", "f1", "roc_auc"])

print(cv_results['test_accuracy'].mean())
print(cv_results['test_f1'].mean())
print(cv_results['test_roc_auc'].mean())

# Analyzing Model Complexity with Learning Curves (BONUS)
def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()

rf_val_params = [["max_depth", [5, 8, 15, 20, 30, None]],
                 ["max_features", [3, 5, 7, "auto"]],
                 ["min_samples_split", [2, 5, 8, 15, 20]],
                 ["n_estimators", [10, 50, 100, 200, 500]]]


rf_model = RandomForestClassifier(random_state=17)

for i in range(len(rf_val_params)):
    val_curve_params(rf_model, X, y, rf_val_params[i][0], rf_val_params[i][1])

print(rf_val_params[0][1])
