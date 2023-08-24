import os
import sys
import pickle
import re
import seaborn as sns
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
from sklearn.neighbors import KNeighborsRegressor #like KNN, but for continuous data
from sklearn import preprocessing
import joblib
from sklearn.metrics import accuracy_score

df= pd.read_csv("Dataset/tracks.csv")
df.columns
df.describe()
df.info()
df.isnull().sum()
df = df.dropna(how='any', axis=0)
df

df[df.duplicated()].sum()

#WE are checking for duplicate values

df['release_date'] = pd.to_datetime(df['release_date'])
df['release_year'] = df['release_date'].dt.year
df['release_month'] = df['release_date'].dt.month_name()
df

number_of_tracks_by_year = df.groupby(df['release_year'])['name'].count().reset_index()
number_of_tracks_by_year['Tracks released'] = number_of_tracks_by_year['name']
# Plotting a line chart:
fig = px.line(number_of_tracks_by_year, x='release_year', y='Tracks released')
fig.update_layout(title="Tracks released throughout the years",title_x=0.5,
                  xaxis_title="Year of release", yaxis_title="Number of tracks")

fig.show()

number_of_tracks_by_month = df.groupby(df['release_month'])['name'].count().reset_index()

plt.figure(figsize=(16,5))
sns.histplot(x = 'release_month', y = 'name', data = number_of_tracks_by_month)
plt.xlabel('Month of release')
plt.ylabel('Number of tracks')
plt.title('Tracks released by month')

plt.show()


plt.figure(figsize=(16,8)) # for covering the entire width
sns.scatterplot(x = 'duration_ms', y = 'popularity', data = df)
plt.xlabel('Duration')
plt.ylabel('Popularity')
plt.title('Relation between Duration and Popularity')


number_of_tracks = df.groupby(df['danceability'])['name'].count().reset_index()
number_of_tracks['popularity'] = number_of_tracks['name']
# Plotting a line chart:
fig = px.line(number_of_tracks, x='danceability', y='popularity')
fig.update_layout(title="comparison of popularity and danceability",title_x=0.5,
                  xaxis_title="danceability", yaxis_title="popularity")

fig.show()


df=df.drop(columns='release_date')
month = {'January': 1,'February': 2 ,'March':3, "April":4, 'May':5, "June":6, 'July': 7, 'August':8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
  
# traversing through dataframe
# Gender column and writing
# values where key matches
df.release_month = [month[item] for item in df.release_month]
print(df)


df_quantitative = df
cols_to_drop = []
for column in df:
    if df[column].dtype == 'object':
        cols_to_drop.append(column)
df_quantitative = df.drop(columns=cols_to_drop)

df_quan_2016_unnormalized = df_quantitative[df_quantitative['release_year']>=2016] #so our songs are more recent
print(f"Working dataset shape: {df_quan_2016_unnormalized.shape}")

df_quan_2016_nm=(df_quan_2016_unnormalized-df_quan_2016_unnormalized.min())/(df_quan_2016_unnormalized.max()-df_quan_2016_unnormalized.min())

df_quan_2016_nm=df_quan_2016_nm.drop(columns='release_year')



np.random.seed(1) #so we can replicate results

#creates a mask, randomly selects 80% of the songs
df_train_full = df_quan_2016_nm.sample(frac=0.8,random_state=1) #random state is a seed value
df_test = df_quan_2016_nm.drop(df_train_full.index)

df_validation = df_train_full.sample(frac=0.2,random_state=2) # create a validation set from training set
df_train = df_train_full.drop(df_validation.index)

# # seperate the data, the Y is what we want to predict
predict = "popularity"
X_train = df_train.drop(columns=[predict])
X_validation = df_validation.drop(columns=[predict])
X_test = df_test.drop(columns=[predict])

Y_train = df_train[[predict]].values.ravel() # .values.ravel() converts column vec. to 1d array
Y_validation = df_validation[[predict]].values.ravel()
Y_test = df_test[[predict]].values.ravel()

def calculate_error(Y_pred, Y_actual):
    error = 0
    for i in range(len(Y_pred)):
        error += abs(Y_pred[i] - Y_actual[i])**2
    return error / len(Y_pred)


# we will run a k nearest neighbours algorithm on this dataset,
# so we will run the algorithm many times to find the best k value
k_errors = [np.inf] # k=0 should have infinite error
for k in range(1,50):
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train, Y_train) 
    Y_val_pred = model.predict(X_validation)
    k_errors.append(calculate_error(Y_val_pred, Y_validation))

plt.scatter(x=range(len(k_errors)), 
            y=k_errors)
plt.xlabel('value of k')
plt.ylabel('error')
plt.title('Error values for different k values on a KNN regressor')
plt.grid(axis='both',alpha=0.5)


plt.show()

k=7
model = KNeighborsRegressor(n_neighbors=k)
model.fit(X_train, Y_train) 
Y_pred = model.predict(X_test)



print(f"Our testing error is {calculate_error(Y_pred, Y_test)}\n\n")

filename = 'trained_model.sav'
pickle.dump(model,open(filename,'wb'))
loaded_model = pickle.load(open('trained_model.sav','rb'))
