import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve,mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

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

df_copy = pd.read_csv("hitters.csv")
df = df_copy.copy()
#######################################################333
###İlk Model
df.dropna(inplace=True)


cat_cols, num_cols, cat_but_car = grab_col_names(df)
df = one_hot_encoder(df, cat_cols, drop_first=True)

y = df['Salary'].astype('int')
X = df.drop("Salary", axis=1).astype('int')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

log = LogisticRegression()
log_model = log.fit(X_train, y_train)

df["Salary"].mean()

# Train RMSE
np.sqrt(mean_squared_error(y_train, log_model.predict(X_train)))

# Train CV
-np.mean(cross_val_score(log_model,
                         X_train,
                         y_train,
                         cv=5,
                         scoring="neg_root_mean_squared_error"))

# Tüm veri CV
-np.mean(cross_val_score(log_model,
                         X,
                         y,
                         cv=5,
                         scoring="neg_root_mean_squared_error"))

np.sqrt(mean_squared_error(y_test, log_model.predict(X_test)))
#######################################################################################33

df = df_copy.copy()
checkDataframe(df)

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

### Hani kolonlarda 0 değerleri var.
### Bölme işleminde paydalara 0 geldiği takdirde 1 ile değiştireceğim.
for i in df.columns:
    print(i,len(df[df[i] == 0]))

# Feature
df['AtBat/CAtBat'] = df['AtBat'] / df['CAtBat']
df['Hits/CHits'] = df['Hits'] / df['CHits']
df['Runs/CRuns'] = df['Runs'] / df['CRuns']
df['Hits/AtBat'] = df['Hits'] / df['AtBat']
df['CHANGE_LEAGUE'] = np.where(df['League'] ==  df['NewLeague'], 1, 0)
df["NEW_AVG_ATBAT"] = df["CAtBat"] / df["Years"]
df["NEW_AVG_HITS"] = df["CHits"] / df["Years"]
df["NEW_AVG_HMRUN"] = df["CHmRun"] / df["Years"]
df["NEW_AVG_RUNS"] = df["CRuns"] / df["Years"]
df["NEW_AVG_RBI"] = df["CRBI"] / df["Years"]
df["NEW_AVG_WALKS"] = df["CWalks"] / df["Years"]
df["NEW_YEAR_CAT"] = pd.qcut(df["Years"], q=4)


#df['Walks/CWalks'] = df['Walks'] / df['CWalks']
df['Walks/CWalks'] = df['Walks'] / np.where(df['CWalks'] == 0, 1, df['CWalks'])
df['Walks/Errors'] = df['Walks'] /np.where(df['Errors'] == 0, 1, df['Errors'])
#df['RBI/CRBI'] = df['RBI'] / df['CRBI']
df['RBI/CRBI'] = df['RBI'] / np.where(df['CRBI'] == 0, 1, df['CRBI'])
df['AtBat/RBI'] = df['AtBat'] / np.where(df['RBI']==0, 1,df['RBI'])
#df['Hits/Runs'] = df['Hits'] / df['Runs']
df['Hits/Runs'] = df['Hits'] / np.where(df['Runs']==0, 1,df['Runs'])
#df['Hits/Errors'] = df['Hits'] / df['Errors']
df['Hits/Errors'] = df['Hits'] / np.where(df['Errors']==0, 1,df['Errors'])

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

df.dropna(inplace=True)


cat_cols, num_cols, cat_but_car = grab_col_names(df)
rare_analyser(df, "Salary", cat_cols)

for col in num_cols:
    num_summary(df, col, plot=False)
for col in num_cols:
    print(col, check_outlier(df, col))
for col in num_cols:
    print(col, replace_with_thresholds(df, col))

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

df1.info()


# Model
y = df['Salary'].astype('int')
X = df.drop("Salary", axis=1).astype('int')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
X_train.info()

# Model:
log = LogisticRegression()
log_model = log.fit(X_train, y_train)
log_model.intercept_
log_model.coef_


# Train RMSE
y_pred = log_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))

# TRAIN RKARE
log_model.score(X_train, y_train)
# TEST RMSE
y_pred = log_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))


# Test RKARE
log_model.score(X_test, y_test)

# Tüm veri CV
-np.mean(cross_val_score(log_model,
                         X,
                         y,
                         cv=5,
                         scoring="neg_root_mean_squared_error"))


-np.mean(cross_val_score(log_model,
                         X_train,
                         y_train,
                         cv=10,
                         scoring="neg_root_mean_squared_error"))



