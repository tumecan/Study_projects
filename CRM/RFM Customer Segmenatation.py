
import datetime as dt
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# 2009-2010 yılı içerisindeki veriler
df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2009-2010")
df = df_.copy()


df.head()
df.shape
df.describe().T



df.isnull().sum()
df.dropna(inplace = True)

## bu şekilde bütün  iadeleri çıkardık.
df = df[~df['Invoice'].str.contains('C', na = False)]

df = df[df['Quantity'] > 0]
df = df[df['Price']>0]


df['TotalPrice']  = df['Quantity'] * df['Price']

###############################################################
# 5. RFM Skorlarının Hesaplanması (Calculating RFM Scores)

## frequency ve recency için max tarih lazım
## analizi yağtığımız tarihi max tarih + 2 gün yapıyoruz.
## Sebebi saat farkları gibi şeyler sorun yaratmasın.
df["InvoiceDate"].max()

today_date = dt.datetime(2010,12,11)

## Müşteri başı

rfm = df.groupby('Customer ID').agg({'InvoiceDate': lambda x:(today_date-x.max()).days,
                                     'Invoice':lambda x:x.nunique(),
                                      'TotalPrice': lambda x:x.sum()})


rfm.head()


rfm.columns = ['recency', 'frequency', 'monetary']

###############################################################
# 5. RFM Skorlarının Hesaplanması (Calculating RFM Scores)

## en küçüğüne 5 en büyüğüne 1
## 1 günce önce gelen ile 40 gün önce geleni kıyaslamak için.
rfm['recency_score'] = pd.qcut(rfm['recency'],5, labels = [5,4,3,2,1])


## en büyüğe 5 en küçüğe 1
## en çok alana 5 veriyoruz.
rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method = 'first'),5,labels=[1,2,3,4,5])


rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

rfm['RFM_SCORE'] = (rfm['recency_score'].astype(str) +
                    rfm["frequency_score"].astype(str) +
                    rfm["monetary_score"].astype(str))

rfm.head()

rfm.describe().T
rfm[rfm["RFM_SCORE"] == "555"].head()


###############################################################
# 6. RFM Segmentlerinin Oluşturulması ve Analiz Edilmesi (Creating & Analysing RFM Segments)

# RFM isimlendirmesi
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'}

rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)

rfm.head()


rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])








