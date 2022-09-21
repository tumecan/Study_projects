

#################################    ÖDEV 2    ##################################


###########      USER --  BASED      #####################

### import os
### print(os.getcwd())


import pandas as pd

def createUsermovie_df():
   ### normalde buraya konulmaz ama scritp olduğu için kullanılabilir. import pandas as pd
    movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1250].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

def chooseUserId(randomUser= True, randomState = 45, userId = None):

 ### Define UserID, Finding Movies which were watched by User
 ## If randomUser True Function will take random user from database
 ## If randomUser NOT True please choose userID and define in userID

 user_movie_df = createUsermovie_df()
 if randomUser:
  random_user = int(pd.Series(user_movie_df.index).sample(1, random_state= randomState).values)
 else:
  random_user = userId
 random_user_df = user_movie_df[user_movie_df.index == random_user]
 movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
 ##print(random_user)
 ##print(random_user_df)
 ##print(movies_watched)
 return random_user,random_user_df,movies_watched,user_movie_df

#
random_user,random_user_df,movies_watched,user_movie_df = chooseUserId()

def userBasedRecommender(percentege = 0.7, corr_value = 0.65, movie_average_rate = 3.7):
 ### Pecentage : The percentage of movies that watched by userid who choosed by YOU :)
 ### Corr_value: The minimum correlation value between user id with other user will be corr_value.
 ### movie_average_rate: The average weighted movie rate will be bigger than value.
 ###random_user, random_user_df, movies_watched ,user_movie_df= chooseUserId()

 movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

 #movies_watched_df = createUsermovie_df()[movies_watched]
 movies_watched_df = user_movie_df[movies_watched]
 user_movie_count = movies_watched_df.T.notnull().sum()
 user_movie_count = user_movie_count.reset_index()

 user_movie_count.columns = ["userId", "movie_count"]
 perc = len(movies_watched) * percentege
 users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]

 final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                       random_user_df[movies_watched]])


 corr_df = final_df.T.corr().unstack().sort_values()
 corr_df = pd.DataFrame(corr_df, columns=["corr"])
 corr_df.index.names = ['user_id_1', 'user_id_2']
 corr_df = corr_df.reset_index()

 top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= corr_value)][
  ["user_id_2", "corr"]].reset_index(drop=True)

 top_users = top_users.sort_values(by='corr', ascending=False)
 top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

 rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')

 top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
 top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']
 top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]

 recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
 recommendation_df = recommendation_df.reset_index()

 movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > movie_average_rate].sort_values("weighted_rating",
                                                                                                    ascending=False)
 movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')

 return movies_to_be_recommend.merge(movie[["movieId", "title"]])[1:6]


user_movie_df = createUsermovie_df()
random_user,random_user_df,movies_watched,user_movie_df = chooseUserId()
userBasedRecommender(percentege = 0.7, corr_value = 0.65, movie_average_rate = 3.7)


##########################################################
###########################################################
###########      ITEM  --  BASED     #####################

movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')

def itemBasedRecommender(movie_name, user_movie_df):
 movie_name = user_movie_df[movie_name]
 return user_movie_df.corrwith(movie_name).sort_values(ascending=False)


def itemBasedChooseMovie(userId = True, user = None,
                         movie_id = False, movieID = None,
                         movieName = None):
 ### If userId  is True then input will user id
 ### If userId is  False and movie_id is True input will movieID
 ### IF userId and movie_id areFalse then input will be movieName
 flag = True
 while flag:
  if userId:
   movieid = rating[(rating["userId"] == user) & (rating["rating"] == 5.0)]. \
               sort_values(by="timestamp", ascending=False)["movieId"][1:2].values[0]
   recommendeMovies = itemBasedRecommender(movie[movie["movieId"] == movieid]["title"].values[0], user_movie_df)
   flag = False
  elif  movie_id :
   recommendeMovies = itemBasedRecommender(movie[movie["movieId"] == movieID]["title"].values[0], user_movie_df)
   flag = False
  else:
   recommendeMovies = itemBasedRecommender(movieName,user_movie_df)
   flag = False
 return recommendeMovies[1:6]


itemBasedChooseMovie(user = 108170)
itemBasedChooseMovie(userId = False, movie_id = True, movieID = 25)
itemBasedChooseMovie(userId = False, movieName = "Matrix, The (1999)")