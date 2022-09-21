
### Content based

import os
print(os.getcwd())


import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

df = pd.read_csv("datasets/the_movies_dataset/movies_metadata.csv", low_memory=False)  # DtypeWarning kapamak icin


df.head()
df.shape
df["overview"].head()

#################################
## Count Vectorizer
from sklearn.feature_extraction.text import CountVectorizer
corpus = ['This is the first document.',
          'This document is the second document.',
          'And this is the third one.',
          'Is this the first document?']


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
vectorizer.get_feature_names()
X.toarray()

#################################
## TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(analyzer='word')
X = vectorizer.fit_transform(corpus)
vectorizer.get_feature_names()
X.toarray()

df['overview'].head()

# Delete words like is, in, on, by, the etc..
tfidf = TfidfVectorizer(stop_words='english')
## Delete NA values.
df['overview'] = df['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(df['overview'])
## (45466, 75827)
## Uqiue words count : 75827
## Document count : 45466
tfidf_matrix.shape

#################################
## Cosine Similarity

## Like Manhattahi ,Euklid etc. methods can be used instead of Cosine Similarty method.
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
cosine_sim.shape
cosine_sim[1]

# Recommendation based on similarty
# Delete Na values
df = df[~df["title"].isna()]

indices = pd.Series(df.index, index=df['title'])

# Remove dublicated values.
indices = indices[~indices.index.duplicated(keep='last')]

# Get Movies id
movie_index = indices["Sherlock Holmes"]

# Find movies cosine similarities
cosine_sim[movie_index]

similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])

## Get most similar movies for Sherlock Holmes
movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
df['title'].iloc[movie_indices]

#### Function of Content Based Recommender

def calculate_cosine_sim(dataframe):
    tfidf = TfidfVectorizer(stop_words='english')
    dataframe['overview'] = dataframe['overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(dataframe['overview'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def content_based_recommender(title, cosine_sim, dataframe):
    dataframe = dataframe[~dataframe["title"].isna()]
    indices = pd.Series(dataframe.index, index=dataframe['title'])
    indices = indices[~indices.index.duplicated(keep='last')]
    movie_index = indices[title]
    similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
    movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
    return dataframe['title'].iloc[movie_indices]

cosine_sim = calculate_cosine_sim(df)
content_based_recommender('Sherlock Holmes', cosine_sim, df)
