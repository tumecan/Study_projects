import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve


pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


sys.path.insert(0,'helpers/')


from eda import *
from data_prep import *

dataframe= pd.read_csv("//titanic.csv")



def titanic_data_preb(dataframe):
    dataframe.columns = [col.upper() for col in dataframe.columns]

    checkDataframe(dataframe)

    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

    num_cols = [col for col in num_cols if "PASSENGERID" not in col]

    for col in cat_cols:
        print(col, cat_summary(dataframe, col))

    for col in num_cols:
        print(col, num_summary(dataframe, col))

    missing_vs_target(dataframe, "SURVIVED", missing_values_table(dataframe, True))

    dataframe['NEW_CABIN'] = dataframe["CABIN"].notnull().astype('int')
    dataframe['EMBARKED'] = dataframe['EMBARKED'].fillna(dataframe['EMBARKED'].mode()[0])

    dataframe['NEW_TITLE'] = dataframe.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)
    dataframe["AGE"] =dataframe["AGE"].fillna(dataframe.groupby("NEW_TITLE")["AGE"].transform("median"))
    dataframe.drop("CABIN", inplace=True, axis=1)

    #family size
    dataframe["NEW_FAMILY_SIZE"] = dataframe["SIBSP"] + dataframe["PARCH"] + 1
    # age_pclass
    dataframe["NEW_AGE_PCLASS"] = dataframe["AGE"] * dataframe["PCLASS"]
    # is alone
    dataframe.loc[((dataframe['SIBSP'] + dataframe['PARCH']) > 0), "NEW_IS_ALONE"] = "NO"
    dataframe.loc[((dataframe['SIBSP'] + dataframe['PARCH']) == 0), "NEW_IS_ALONE"] = "YES"
    # age level
    dataframe.loc[(dataframe['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
    dataframe.loc[(dataframe['AGE'] >= 18) & (dataframe['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
    dataframe.loc[(dataframe['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'
    # sex x age
    dataframe.loc[(dataframe['SEX'] == 'male') & (dataframe['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
    dataframe.loc[
        (dataframe['SEX'] == 'male') & ((dataframe['AGE'] > 21) & (dataframe['AGE']) <= 50), 'NEW_SEX_CAT'] = 'maturemale'
    dataframe.loc[(dataframe['SEX'] == 'male') & (dataframe['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (dataframe['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (
                (dataframe['AGE'] > 21) & (dataframe['AGE']) <= 50), 'NEW_SEX_CAT'] = 'maturefemale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (dataframe['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'

    remove_cols = ["TICKET", "NAME"]
    dataframe.drop(remove_cols, inplace=True, axis=1)

    for col in num_cols:
        replace_with_thresholds(dataframe, col)

    dataframe = dataframe.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)


    label_col = [col for col in dataframe.columns if
                     dataframe[col].dtype not in [int, float] and dataframe[col].nunique() == 2]
    for col in label_col:
        dataframe = label_encoder(dataframe, col)

    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
    ## Rare Encoding

    cat_cols
    rare_analyser(dataframe, 'SURVIVED', cat_cols)

    dataframe = rare_encoder(dataframe, 0.01, cat_cols)

    ## One Hot Encoder
    ohe_cols = [col for col in dataframe.columns if 10 >= dataframe[col].nunique() > 2]
    dataframe = one_hot_encoder(dataframe, ohe_cols)
    dataframe.head()

    scaler = StandardScaler()
    dataframe[num_cols] = scaler.fit_transform(dataframe[num_cols])
    dataframe[num_cols].head()

    return dataframe

def load():
    data = pd.read_csv("titanic.csv")
    return data

df = load()


titanic_new_data = titanic_data_preb(df)


y = titanic_new_data["SURVIVED"]
X = titanic_new_data.drop(["PASSENGERID", "SURVIVED"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

log_model = LogisticRegression().fit(X_train, y_train)
log_model.intercept_
log_model.coef_


#train
y_pred = log_model.predict(X_train)

# Başarı skorları:
print(classification_report(y_train, y_pred))

# ROC AUC
y_prob = log_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_prob)


# Model Validation: 5-Fold Cross Validation

cv_results = cross_validate(log_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])


cv_results['test_accuracy'].mean()
# Accuracy: 0.8305

cv_results['test_precision'].mean()
# Precision: 0.7951

cv_results['test_recall'].mean()
# Recall: 0.7540

cv_results['test_f1'].mean()
# F1-score: 0.7725

cv_results['test_roc_auc'].mean()
# AUC: 0.8688
