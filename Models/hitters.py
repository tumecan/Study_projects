import os
print(os.getcwd())
os.chdir(r'\Users\tumec\PycharmProjects\dsmlbc6')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve,mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier,AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.simplefilter(action='ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from warnings import filterwarnings
filterwarnings('ignore')


pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


sys.path.insert(0,'helpers/')

from eda import grab_col_names, checkDataframe,cat_summary,num_summary, \
    high_correlated_cols,sweetviz_analiz
from data_prep import outlier_thresholds,replace_with_thresholds,check_outlier,\
    grab_outliers,remove_outlier,missing_values_table,missing_vs_target,label_encoder,one_hot_encoder,\
    rare_analyser,rare_encoder

df_copy = pd.read_csv("datasets/hafta7/hitters.csv")

df = df_copy.copy()

######################################################
# 1. Exploratory Data Analysis
checkDataframe(df)
grab_col_names(df)


df_delete = df[(df['Salary'] > 1300) & (df['Years'] < 5)]
df[(df['Salary'] > 1300)]
## Oyuncuların sezon sonu oynadığı lig hakkında bilgi
df["League"].value_counts()
## Oyuncuların sezon başında oynadığı lig hakkında bilgi
df["NewLeague"].value_counts()
## Oyuncuların Pozisyonları hakkında bilghi vermektedir
df["Division"].value_counts()

# Oynadığı Lige göre maaş ortalaması
df.groupby("League").agg({"Salary": "mean"})
df.groupby("NewLeague").agg({"Salary": "mean"})
df.groupby("Division").agg({"Salary": "mean"})
# Yıl ve Lige göre ortalama maaş
df.groupby(["League","Years"]).agg({"Salary": "mean"})
df.groupby(["Division","Years"]).agg({"Salary": "mean"})
higf_correlated_list = high_correlated_cols(df, True)

## Just salary
missing_values_table(df)


######################################################
# 2. Datapreprocesssing and Feature Engineering

### Hani kolonlarda 0 değerleri var.
### Bölme işleminde paydalara 0 geldiği takdirde 1 ile değiştireceğim.
for i in df.columns:
    print(i,len(df[df[i] == 0]))

# Feature
df['AtBat/CAtBat'] = df['AtBat'] / df['CAtBat']
df['Hits/CHits'] = df['Hits'] / df['CHits']
df['Runs/CRuns'] = df['Runs'] / df['CRuns']
df['Runs/Hits'] = df['Runs'] / df['Hits']
df['CRuns/CHits'] = df['CRuns'] / df['CHits']
df['Hits/AtBat'] = df['Hits'] / df['AtBat']
#df['CHANGE_LEAGUE'] = np.where(df['League'] ==  df['NewLeague'], 1, 0)
df["NEW_AVG_ATBAT"] = df["CAtBat"] / df["Years"]
df["NEW_AVG_HITS"] = df["CHits"] / df["Years"]
df["NEW_AVG_HMRUN"] = df["CHmRun"] / df["Years"]
df["NEW_AVG_RUNS"] = df["CRuns"] / df["Years"]
df["NEW_AVG_RBI"] = df["CRBI"] / df["Years"]
df["NEW_AVG_WALKS"] = df["CWalks"] / df["Years"]
df["NEW_YEAR_CAT"] = pd.qcut(df["Years"], q=4)
checkDataframe(df)
df[(df['Salary'] > 1200) & (df['NEW_AVG_ATBAT'] < 332)]
df[(df['Salary'] > 1200) & (df['NEW_AVG_HITS'] < 86)]
df[(df['Salary'] > 1200) & (df['NEW_AVG_HMRUN'] < 6)]
df[(df['Salary'] > 1200) & (df['NEW_AVG_RUNS'] < 43)]
df[(df['Salary'] > 1200) & (df['NEW_AVG_RBI'] < 36)]
df[(df['Salary'] > 1200) & (df['NEW_AVG_WALKS'] < 27)]
df= df[df.index != 217]

"""
df['Walks/CWalks'] = df['Walks'] / np.where(df['CWalks'] == 0, 0.5, df['CWalks'])
df['Walks/Errors'] = df['Walks'] /np.where(df['Errors'] == 0, 0.5, df['Errors'])

df['RBI/CRBI'] = df['RBI'] / np.where(df['CRBI'] == 0, 0.5, df['CRBI'])
df['AtBat/RBI'] = df['AtBat'] / np.where(df['RBI']==0, 0.5,df['RBI'])

df['Hits/Runs'] = df['Hits'] / np.where(df['Runs']==0, 0.5,df['Runs'])

df['Hits/Errors'] = df['Hits'] / np.where(df['Errors']==0, 0.5,df['Errors'])
df["Assists/PutOuts"] = df["Assists"] / np.where(df['PutOuts'] == 0, 0.5, df['PutOuts'])
"""
"""
df['Walks/CWalks'] = df['Walks'] / np.where(df['CWalks'] == 0, 000.1, df['CWalks'])
df['Walks/Errors'] = df['Walks'] /np.where(df['Errors'] == 0, 000.1, df['Errors'])
df['RBI/CRBI'] = df['RBI'] / np.where(df['CRBI'] == 0, 000.1, df['CRBI'])
df['AtBat/RBI'] = df['AtBat'] / np.where(df['RBI']==0, 000.1,df['RBI'])
df['Hits/Runs'] = df['Hits'] / np.where(df['Runs']==0, 000.1,df['Runs'])
df['Hits/Errors'] = df['Hits'] / np.where(df['Errors']==0, 0.0001,df['Errors'])
"""

for i in df.columns:
    print(i,len(df[df[i] == 0]))

missing_values_table(df)
####msno.matrix(df.sample(250))
df.groupby(['League', 'NewLeague','Division','NEW_YEAR_CAT']).agg({'Salary': "mean"})
df["Salary"] = df["Salary"].fillna(df.groupby(['League', 'NewLeague','Division','NEW_YEAR_CAT'])["Salary"].transform("median"))
#df["NEW_SALARY_CAT"] = pd.qcut(df["Salary"], q=10)

df.dropna(inplace=True)


cat_cols, num_cols, cat_but_car = grab_col_names(df)
rare_analyser(df, "Salary", cat_cols)

for col in num_cols:
    num_summary(df, col, plot=False)
for col in num_cols:
    print(col, check_outlier(df, col, q1= 0.1, q3= 0.9))
for col in num_cols:
    print(col, replace_with_thresholds(df, col, q1= 0.1, q3= 0.9))

df.describe().T

"""
def target_correlation_matrix(dataframe, corr_th=0.7, target="Salary"):

    corr = dataframe.corr()
    corr_th = corr_th
    try:
        filter = np.abs(corr[target]) > corr_th
        corr_features = corr.columns[filter].tolist()
        sns.clustermap(dataframe[corr_features].corr(), annot=True, fmt=".2f")
        plt.show()
        return corr_features
    except:
        print("Yüksek threshold değeri, corr_th değerinizi düşürün!")


target_correlation_matrix(df, corr_th=0.5, target="Salary")
"""
df = one_hot_encoder(df, cat_cols, drop_first=True)

df1 = df.copy()

#df1.info()

######################################################
# 3. Base Model

y = df1['Salary'].astype('int')
X = df1.drop("Salary", axis=1).astype('int')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
X_train.info()


regressors = [("CART", DecisionTreeClassifier()),
               ("RF", RandomForestClassifier()),
               ('GBM', GradientBoostingClassifier()),
               ('XGBoost', XGBClassifier(use_label_encoder=False)),
               ('LightGBM', LGBMClassifier())]

for name, regressor in regressors:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")


######################################################
# 4. Automated Hyperparameter Optimization



cart_params = {'max_depth': range(1, 29),
               "min_samples_split": range(2, 30),
               "min_weight_fraction_leaf":[0,0.1,0.2,0.3,0.4,0.5]}

rf_params = {"max_depth": [5, 8 ,15,None],
             "max_features": [5,7,15, "auto"],
             "min_samples_split": [5,8,15,20],
             #"criterion": ["gini", "entropy"],
             "n_estimators": [200,500,600,700,750]}

gbm_params = {"learning_rate": [0.01, 0.1,0.02],
              "max_depth": range(1, 20),
              "n_estimators": [200,500,600,700,750],
              "subsample": [0.4,0.6, 0.8, 1]}

xgboost_params = {"learning_rate": [0.01, 0.1,0.02],
                  "max_depth": [5, 8, 12, 20],
                  "n_estimators": [200,500,600,700,750],
                   "colsample_bytree": [0.5, 0.7,0.8, 1]}

lgbm_params = {"learning_rate": [0.01, 0.1,0.02],
               "n_estimators": [200,500,600,700,750],
                "max_depth": [5, 8, 12, 20],
               "colsample_bytree": [0.5, 0.7,0.8, 1]}

regressors = [("CART", DecisionTreeClassifier(),cart_params),
               ("RF", RandomForestClassifier(),rf_params),
               ('GBM', GradientBoostingClassifier(),gbm_params),
               ('XGBoost', XGBClassifier(),xgboost_params),
               ('LightGBM', LGBMClassifier(),lgbm_params)]

def hyperparameter_optimization(X,y,cv = 5, score = 'roc_auc'):
    for name, regressor, params in regressors:
        print(f"########## {name} ##########")
        rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=cv, scoring=score)))
        #rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=cv, scoring=scoring)))
        print(f"RMSE: {round(rmse, 4)} ({name}) ")

        gs_best = GridSearchCV(regressor, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)

        final_model = regressor.set_params(**gs_best.best_params_)
        #y_pred = final_model.fit(X_train, y_train).predict(X_test)
        print("MSE of 'y_test -- y_pred' ")
        #print(np.sqrt(mean_squared_error(y_test, y_pred)))
        rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=cv, scoring=score)))
        #rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=10, scoring=scoring)))
        print(f"RMSE (After): {round(rmse, 4)} ({name}) ")

        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

        best_models[name] = final_model
    return best_models

best_models =  hyperparameter_optimization(X,y, cv = 10,score =  "neg_mean_squared_error" )

######################################################
# 5. Stacking and  Ensemble Learning

def voting_classifier(best_models, X,y, cv = 3):
    voting_clf = VotingClassifier(
        estimators=[('CART', best_models["CART"]),
                    ('RF', best_models["RF"]),
                    ('GBM', best_models["GBM"]),
                    ('XGBoost', best_models["XGBoost"]),
                    ('LightGBM', best_models["LightGBM"])],
        voting='soft')

    voting_clf.fit(X, y)

    cv_results = cross_validate(voting_clf, X, y, cv=cv, scoring=["accuracy", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1_score: {cv_results['test_f1'].mean()}")
    print(f"Roc_Auc: {cv_results['test_roc_auc'].mean()}")

    return voting_clf

voting_clf = voting_classifier(best_models,X,y,cv = 10)

######################################################
# 6. Prediction for a New Observation

X.columns

random_user = X.sample(1, random_state=451)
voting_clf.predict(random_user)

import pickle

pickle.dump(open(voting_clf, "voting_clf.pkl", "wb"))

pickle.dump(voting_clf, open('voting_clf.pkl', 'wb'))




######################################################
# 4. Automated Hyperparameter Optimization


cart_params = {'max_depth': range(1, 30),
               "min_samples_split":  [5,8,15,20,30],
               "min_weight_fraction_leaf":[0,0.1,0.3,0.4,0.5]}

rf_params = {"max_depth": [5, 9 ,15,20,None],
             "max_features": [7,15,25, "auto"],
             "min_samples_split": [5,8,15,20],
             #"criterion": ["gini", "entropy"],
             "n_estimators": [200,500,700,1000]}

gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": range(1, 20),
              "n_estimators": [200,500,700,1000],
              "subsample": [0.4,0.6, 0.8, 1]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [1,2,5,7,15,20],
                  "n_estimators": [200,500,700,1000],
                   "colsample_bytree": [0.5, 0.6,0.7,0.8, 0.9, 1]}

lgbm_params = {"learning_rate": [0.01, 0.02, 0.03, 0.05, 0.07, 0.1],
               "n_estimators": [200,500,700,1000],
                "max_depth": [1,2,5,7,15,20],
               "colsample_bytree": [0.5, 0.6,0.7,0.8, 0.9, 1]}

regressors = [("CART", DecisionTreeClassifier(),cart_params),
               ("RF", RandomForestClassifier(),rf_params),
               ('GBM', GradientBoostingClassifier(),gbm_params),
               ('XGBoost', XGBClassifier(),xgboost_params),
               ('LightGBM', LGBMClassifier(),lgbm_params)]

def hyperparameter_optimization(X,y,cv = 5, score = 'roc_auc'):
    best_models = {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
    for name, regressor, params in regressors:
        print(f"########## {name} ##########")
        rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=cv, scoring=score)))
        #rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=cv, scoring=scoring)))
        print(f"RMSE: {round(rmse, 4)} ({name}) ")

        gs_best = GridSearchCV(regressor, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)

        final_model = regressor.set_params(**gs_best.best_params_)
        y_pred = final_model.fit(X_train, y_train).predict(X_test)
        print("MSE of 'y_test -- y_pred' ")
        print(np.sqrt(mean_squared_error(y_test, y_pred)))
        rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=cv, scoring=score)))
        #rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=10, scoring=scoring)))
        print(f"RMSE (After): {round(rmse, 4)} ({name}) ")

        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

        best_models[name] = final_model
    return best_models

best_models =  hyperparameter_optimization(X,y, cv = 10,score =  "neg_mean_squared_error" )


