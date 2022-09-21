### ARL Recommender

import os
print(os.getcwd())



import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

## Display whole output in screen
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)


df_ = pd.read_excel('datasets/online_retail_II.xlsx', sheet_name='Year 2010-2011')
df = df_.copy()

def checkDataframe(dataframe,rowNumber = 10):

    print('####  Columns name #####')
    print(dataframe.columns)
    print('####  Shape   #####')
    print(dataframe.shape)
    print('####  First 5 Rows #####')
    print(dataframe.head(rowNumber))
    print('####  Last 5 Rows #####')
    print(dataframe.head(rowNumber))
    print('####  Info #####')
    print(dataframe.info())
    print('####  Describe #####')
    print(dataframe.describe().T)
    print('####  Null Count Each Columns #####')
    print(dataframe.isnull().sum())
    print('####  Categorical Columns Names #####')
    ###### Not include cardinal columns
    cat_cols = [col for col in dataframe.columns if (dataframe[col].nunique() < 20 and dataframe[col].dtypes == "O")
                or (dataframe[col].nunique() < 20 and dataframe[col].dtypes != "O")]
    print(cat_cols)
    ##### Not include cardinal columns
    print('####   Numerical Columns Names #####')
    num_cols = [col for col in dataframe.columns if (dataframe[col].nunique() > 20 and dataframe[col].dtypes != "O")]
    print(num_cols)


checkDataframe(df)

## Number of uniqe product description number
df["Description"].nunique()
## Count of each product count
df["Description"].value_counts().head()
## Sum of quantity fot each description
df.groupby("Description").agg({"Quantity": "sum"}).head()
## Number of unique invoice number
df["Invoice"].nunique()
## Minimum quantity number
df.sort_values("Quantity", ascending=True).head()
## The Invoice which include C is represent to returs products..
df[df["Invoice"].str.contains("C", na=False)].head()
##Most expencie prodıct
df.sort_values("Price", ascending=False).head()
## Order number from each country
df["Country"].value_counts()


## Thresholds
def outlierThresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replaceWithThresholds(dataframe, variable):
    low_limit, up_limit = outlierThresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retailDataPrep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replaceWithThresholds(dataframe, "Quantity")
    replaceWithThresholds(dataframe, "Price")
    return dataframe

df = retailDataPrep(df)

df_fr = df[df['Country'] == "France"]

def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)

fr_inv_pro_df = create_invoice_product_df(df_fr, id = True)
fr_inv_pro_df.head()

def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)


##check_multiple_Id(df_fr, 23166,23084)


## Create Rules of probabilty of products in a basket.
def create_rules(dataframe, id=True, country="France"):
    dataframe = dataframe[dataframe['Country'] == country]
    dataframe = create_invoice_product_df(dataframe, id)
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules

rules = create_rules(df)

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list= []

    for i, product in sorted_rules["antecedents"].items():
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))
    recommendation_list = list({item for item_list in recommendation_list for item in item_list})

    return recommendation_list[:rec_count]

def check_multiple_Id(dataframe, stock_codes = list):
    for stock_code in stock_codes:
        product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
        print(stock_code,product_name)

check_multiple_Id(df, arl_recommender(rules, 23084, 2)) #['PACK OF 20 SKULL PAPER NAPKINS']

