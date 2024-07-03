"""
###################################################
# PROJECT: SALARY PREDICTİON WITH MACHINE LEARNING
###################################################

# İş Problemi

# Maaş bilgileri ve 1986 yılına ait kariyer istatistikleri paylaşılan beyzbol
# oyuncularının maaş tahminleri için bir makine öğrenmesi projesi gerçekleştirilebilir mi?

# Veri seti hikayesi

# Bu veri seti orijinal olarak Carnegie Mellon Üniversitesi'nde bulunan StatLib kütüphanesinden alınmıştır.
# Veri seti 1988 ASA Grafik Bölümü Poster Oturumu'nda kullanılan verilerin bir parçasıdır.
# Maaş verileri orijinal olarak Sports Illustrated, 20 Nisan 1987'den alınmıştır.
# 1986 ve kariyer istatistikleri, Collier Books, Macmillan Publishing Company, New York tarafından yayınlanan
# 1987 Beyzbol Ansiklopedisi Güncellemesinden elde edilmiştir.


# AtBat: 1986-1987 sezonunda bir beyzbol sopası ile topa yapılan vuruş sayısı
# Hits: 1986-1987 sezonundaki isabet sayısı
# HmRun: 1986-1987 sezonundaki en değerli vuruş sayısı
# Runs: 1986-1987 sezonunda takımına kazandırdığı sayı
# RBI: Bir vurucunun vuruş yaptıgında koşu yaptırdığı oyuncu sayısı
# Walks: Karşı oyuncuya yaptırılan hata sayısı
# Years: Oyuncunun major liginde oynama süresi (sene)
# CAtBat: Oyuncunun kariyeri boyunca topa vurma sayısı
# CHits: Oyuncunun kariyeri boyunca yaptığı isabetli vuruş sayısı
# CHmRun: Oyucunun kariyeri boyunca yaptığı en değerli sayısı
# CRuns: Oyuncunun kariyeri boyunca takımına kazandırdığı sayı
# CRBI: Oyuncunun kariyeri boyunca koşu yaptırdırdığı oyuncu sayısı
# CWalks: Oyuncun kariyeri boyunca karşı oyuncuya yaptırdığı hata sayısı
# League: Oyuncunun sezon sonuna kadar oynadığı ligi gösteren A ve N seviyelerine sahip bir faktör
# Division: 1986 sonunda oyuncunun oynadığı pozisyonu gösteren E ve W seviyelerine sahip bir faktör
# PutOuts: Oyun icinde takım arkadaşınla yardımlaşma
# Assits: 1986-1987 sezonunda oyuncunun yaptığı asist sayısı
# Errors: 1986-1987 sezonundaki oyuncunun hata sayısı
# Salary: Oyuncunun 1986-1987 sezonunda aldığı maaş(bin uzerinden)
# NewLeague: 1987 sezonunun başında oyuncunun ligini gösteren A ve N seviyelerine sahip bir faktör
"""

###################################################
# GÖREV: Veri ön işleme ve özellik mühendisliği tekniklerini kullanarak maaş tahmin modeli geliştiriniz.
###################################################

############################################
# Gerekli Kütüphane ve Fonksiyonlar
############################################

import warnings
import pandas as pd
import missingno as msno
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler

from helpers.eda import *
from helpers.data_prep import *
from helpers.helpers import *
from helpers.feature import *
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

# Tum Base Modeller
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

from pandas.core.common import SettingWithCopyWarning
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

df = pd.read_csv("hitters.csv")

############################################
# EDA ANALIZI
############################################

check_df(df)

# Bağımlı değişkende 59 tane NA var!
# CAtBat, CHits outlier olabilir.

df.shape
# BAĞIMLI DEĞİŞKEN ANALİZİ

df["Salary"].describe()
sns.distplot(df.Salary)
plt.show()

sns.boxplot(df["Salary"])
plt.show()

# KATEGORİK VE NUMERİK DEĞİŞKENLERİN SEÇİLMESİ
cat_cols, cat_but_car, num_cols, num_but_cat= grab_col_names(df)

# KATEGORİK DEĞİŞKEN ANALİZİ
rare_analyser(df, "Salary", 0.05)

# SAYISAL DEĞİŞKEN ANALİZİ
for col in num_cols:
    num_summary(df, col, plot=False)

# AYKIRI GÖZLEM ANALİZİ
for col in num_cols:
    print(col, check_outlier(df, col, quartile_1=0.1, quartile_3=0.9))

# 1300 den sonraki değerleri veri setinden çıkartıyorum.
df['Salary'].describe([0.7,0.8,0.9,0.95])
print(df.shape)
df = df[(df['Salary'] < 1350) | (df['Salary'].isnull())]  # Eksik değerleri de istiyoruz.

print(df.shape)
sns.distplot(df.Salary)
plt.show()

# AYKIRI DEĞERLERİ BASKILAMA
new_num_cols=[col for col in num_cols if col!="Salary"]
for col in new_num_cols:
    if check_outlier(df, col, quartile_1=0.05, quartile_3=0.95):
        replace_with_thresholds(df, col, quartile_1=0.05, quartile_3=0.95)

# EKSİK GÖZLEM ANALİZİ

missing_values_table(df)  # Salary bağımlı değişkeninde 59 Eksik Gözlem bulunmakta. Bunları çıkartmak bir çözüm yolu olabilir.


# KORELASYON ANALİZİ

def target_correlation_matrix(dataframe, corr_th=0.5, target="Salary"):
    """
    Bağımlı değişken ile verilen threshold değerinin üzerindeki korelasyona sahip değişkenleri getirir.
    :param dataframe:
    :param corr_th: eşik değeri
    :param target:  bağımlı değişken ismi
    :return:
    """
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

############################################
# VERİ ÖNİŞLEME
############################################

df['NEW_HitRatio'] = df['Hits'] / df['AtBat']
df['NEW_RunRatio'] = df['HmRun'] / df['Runs']
df['NEW_CHitRatio'] = df['CHits'] / df['CAtBat']
df['NEW_CRunRatio'] = df['CHmRun'] / df['CRuns']

df['NEW_Avg_AtBat'] = df['CAtBat'] / df['Years']
df['NEW_Avg_Hits'] = df['CHits'] / df['Years']
df['NEW_Avg_HmRun'] = df['CHmRun'] / df['Years']
df['NEW_Avg_Runs'] = df['CRuns'] / df['Years']
df['NEW_Avg_RBI'] = df['CRBI'] / df['Years']
df['NEW_Avg_Walks'] = df['CWalks'] / df['Years']


# One Hot Encoder

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


df = one_hot_encoder(df, cat_cols, drop_first=True)

############################################
# MODELLEME
############################################

df.dropna(inplace=True)  # Salarydeki eksik değerleri çıkartma

y = df['Salary']
X = df.drop("Salary", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
model=LinearRegression()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(model, rmse)   #222


rmse = np.mean(np.sqrt(-cross_val_score(model, X, y, cv=10, scoring="neg_mean_squared_error")))
rmse  #268



model2=GradientBoostingRegressor(random_state=17)

model2.fit(X_train, y_train)
y_pred = model2.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(model, rmse) #164

model2=GradientBoostingRegressor(random_state=17)
rmse = np.mean(np.sqrt(-cross_val_score(model2, X, y, cv=10, scoring="neg_mean_squared_error")))
rmse  #190



params = {"learning_rate": [0.01],
                   "n_estimators": [400,450,500,525,550],
                   "subsample": [0.4,0.5,0.6, 0.7],
                    "max_depth":[1,2,3]}

print(f"########## GBM ##########")

gs_best = GridSearchCV(model2, params, cv=3, n_jobs=-1, verbose=False).fit(X, y)
final_model = model2.set_params(**gs_best.best_params_)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))  #157

np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=10, scoring="neg_mean_squared_error")))  #187
#187



