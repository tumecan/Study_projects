
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import missingno as msno
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split, cross_validate

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


sys.path.insert(0,'helpers/')

from eda import grab_col_names, checkDataframe,cat_summary,num_summary, \
    high_correlated_cols,sweetviz_analiz
from data_prep import outlier_thresholds,replace_with_thresholds,check_outlier,\
    grab_outliers,remove_outlier,missing_values_table,missing_vs_target,label_encoder,one_hot_encoder,\
    rare_analyser,rare_encoder


df_main =pd.read_csv("diabetes.csv")
df = df_main.copy()
checkDataframe(df)

# Hedef değişkenin sınıfları ve frekansları:
df['Outcome'].value_counts()

# Frekanslar görsel olarak
sns.countplot(x= 'Outcome', data= df)
plt.show()

# Hedef Frekans Oranları
df['Outcome'].value_counts()/len(df)

## Yüksek korelasyon yok.
high_correlated_cols(df)


def plot_numerical_col(dataframe, numerical_col):
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    plt.show()


cols = [col for col in df.columns if "Outcome" not in col]

for col in cols:
     plot_numerical_col(df, col)

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in cols:
    target_summary_with_num(df, "Outcome", col)

## Without sex data pregnancies can misdirect to model.

#################
## Missing Values

# Columns that min values can not be zero.
# 0 will be Nan values.
missing_zero = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']

for i in missing_zero:
    df[i] = df[i].replace(0, np.nan)

# Check min values
df.describe().T

# Chacek Nan Values


missing_values_table(df)

for i in missing_zero:
    df[i] = df[i].replace(np.nan, df[i].median())

missing_values_table(df)

## Outlier

for col in cols:
    print(col, check_outlier(df, col))

replace_with_thresholds(df, "Insulin")
replace_with_thresholds(df, "SkinThickness")

for col in cols:
    print(col, check_outlier(df, col))

## Feature Extraction  and Interaction


high_correlated_cols(df, True)

df.describe().T

def diabetes_feature(dataframe):
# Glucose
    dataframe.loc[dataframe['Glucose'] < 71, 'NEW_GLUCOSE'] = 'Hipoglisemi'
    dataframe.loc[(dataframe['Glucose'] >= 71) & (dataframe['Glucose'] <= 100), 'NEW_GLUCOSE'] = 'Normal'
    dataframe.loc[(dataframe['Glucose'] >= 101) & (dataframe['Glucose'] <= 125), 'NEW_GLUCOSE'] = 'Prediabetes'
    dataframe.loc[dataframe['Glucose'] > 125, 'NEW_GLUCOSE'] = 'Probablydiabet'

# BloodPressure
    dataframe.loc[dataframe['BloodPressure'] < 80, 'NEW_BLOOD_PRE'] = 'Optimal'
    dataframe.loc[(dataframe['BloodPressure'] >= 80) & (dataframe['BloodPressure'] <= 84), 'NEW_BLOOD_PRE'] = 'Normal'
    dataframe.loc[(dataframe['BloodPressure'] >= 85) & (dataframe['BloodPressure'] <= 89), 'NEW_BLOOD_PRE'] = 'High_normal'
    dataframe.loc[(dataframe['BloodPressure'] >= 90) & (dataframe['BloodPressure'] <= 120), 'NEW_BLOOD_PRE'] = 'Grade_1_hypertension'
    dataframe.loc[dataframe['BloodPressure'] >= 120, 'NEW_BLOOD_PRE'] = 'Run_Hospital'

# BMI:
    dataframe.loc[dataframe['BMI'] < 18.5, 'NEW_BMI'] = 'Underweight'
    dataframe.loc[(dataframe['BMI'] >= 18.5) & (dataframe['BMI'] < 25.0), 'NEW_BMI'] = 'Normal'
    dataframe.loc[(dataframe['BMI'] >= 25.0) & (dataframe['BMI'] < 30.0), 'NEW_BMI'] = 'Overweight'
    dataframe.loc[(dataframe['BMI'] >= 30.0) & (dataframe['BMI'] < 40.0), 'NEW_BMI'] = 'Obese'
    dataframe.loc[dataframe['BMI'] >= 40.0, 'NEW_BMI'] = 'Morbid_Obese'

# Age:
    dataframe.loc[dataframe['Age'] < 24, 'NEW_AGE'] = 'Young'
    dataframe.loc[(dataframe['Age'] >= 24) & (dataframe['Age'] < 41), 'NEW_AGE'] = 'Mature'
    dataframe.loc[(dataframe['Age'] >= 41) & (dataframe['Age'] < 55), 'NEW_AGE'] = 'Senior'
    dataframe.loc[dataframe['Age'] >= 55, 'NEW_AGE'] = 'Elder'

# Age and Glucose
    dataframe.loc[(dataframe["Glucose"] < 70) & ((dataframe["Age"] >= 21) & (dataframe["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "lowmature"
    dataframe.loc[(dataframe["Glucose"] < 70) & (dataframe["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "lowsenior"

    dataframe.loc[((dataframe["Glucose"] >= 70) & (dataframe["Glucose"] < 100)) & (
            (dataframe["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "normalmature"
    dataframe.loc[((dataframe["Glucose"] >= 70) & (dataframe["Glucose"] < 100)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "normalsenior"

    dataframe.loc[((df["Glucose"] >= 100) & (dataframe["Glucose"] <= 125)) & (
            (dataframe["Age"] >= 21) & (dataframe["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "hiddenmature"
    dataframe.loc[
        ((dataframe["Glucose"] >= 100) & (dataframe["Glucose"] <= 125)) & (dataframe["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "hiddensenior"

    dataframe.loc[(dataframe["Glucose"] > 125) & ((dataframe["Age"] >= 21) & (dataframe["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "highmature"
    dataframe.loc[(dataframe["Glucose"] > 125) & (dataframe["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "highsenior"

# Age and BMI
    dataframe.loc[(dataframe["BMI"] < 18.5) & ((dataframe["Age"] >= 21) & (dataframe["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "underweightmature"
    dataframe.loc[(dataframe["BMI"] < 18.5) & (dataframe["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "underweightsenior"

    dataframe.loc[((dataframe["BMI"] >= 18.5) & (dataframe["BMI"] < 25)) & (
            (dataframe["Age"] >= 21) & (dataframe["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "healthymature"
    dataframe.loc[((dataframe["BMI"] >= 18.5) & (dataframe["BMI"] < 25)) & (dataframe["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "healthysenior"

    dataframe.loc[((dataframe["BMI"] >= 25) & (dataframe["BMI"] < 30)) & (
            (dataframe["Age"] >= 21) & (dataframe["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "overweightmature"
    dataframe.loc[((dataframe["BMI"] >= 25) & (dataframe["BMI"] < 30)) & (dataframe["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "overweightsenior"

    dataframe.loc[(dataframe["BMI"] > 18.5) & ((dataframe["Age"] >= 21) & (dataframe["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "obesemature"
    dataframe.loc[(dataframe["BMI"] > 18.5) & (dataframe["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "obesesenior"

# Insulin
    dataframe.loc[dataframe['Insulin'] <= 150, 'NEW_INSULIN'] = 'Normal'
    dataframe.loc[dataframe['Insulin'] > 150, 'NEW_INSULIN'] = 'High'

    return dataframe


diabetes_feature(df)

###################
# MinMaxScaler

num_col= [col for col in df.columns if df[col].dtypes != "O" and col !='Outcome']

for i in num_col:
    df[i]=MinMaxScaler().fit_transform(df[[i]])

df.head()

df.columns = [col.upper() for col in df.columns]

# Label Encoding

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)

df.head()

df = pd.get_dummies(df, drop_first=True)

df.head()



### Model
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state= 42)

log_model = LogisticRegression().FOR

log_model.intercept_
log_model.coef_


# Tahmin
y_pred = log_model.predict(X)

y_pred[0:10]
y[0:10]

#train
y_pred = log_model.predict(X_train)

# Başarı skorları:
print(classification_report(y_train, y_pred))

# ROC AUC
y_prob = log_model.predict_proba(X_train)[:, 1]
print(classification_report(y_train, y_pred))
roc_auc_score(y_train, y_prob)

# test
y_pred = log_model.predict(X_test)


# ROC AUC
y_prob = log_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_prob)

# ROC Curve
plot_roc_curve(log_model, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()

# Model Validation: 5-Fold Cross Validation

cv_results = cross_validate(log_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])


cv_results['test_accuracy'].mean()
# Accuracy: 0.7734

cv_results['test_precision'].mean()
# Precision: 0.7086

cv_results['test_recall'].mean()
# Recall: 0.6230

cv_results['test_f1'].mean()
# F1-score: 0.6542

cv_results['test_roc_auc'].mean()
# AUC: 0.8378