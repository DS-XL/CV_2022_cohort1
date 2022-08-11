# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

F1 = "./datasets/MeetFresh_menu_drink_1.csv"
F2 = "./datasets/MeetFresh_menu_food_1.csv"
F3 = "./datasets/survey.csv"

# define the threshold for the cosine similarity
T = 0.9980

# df1 = pd.read_csv(F1)
df2 = pd.read_csv(F2)
df3 = pd.read_csv(F3)

# Preprocessing
df3['gender'].replace(['F', 'M'], [0, 1], inplace=True)
df3['ethnicity'].replace(['Asian', 'White','Black','Hispanic'], [1,2,3,4], inplace=True)
df3['hot or cold'].replace(['H', 'C'], [0,1], inplace=True)

# drop unused col
# Note: this step is needed since we don't ask these three features from the user's input
df3_drop = df3.drop(['current state','Favorite Dish ','user_id'], axis=1)

# raw_input = [0, 24, 1, 5, 5, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# input = np.array([raw_input], dtype=np.int32)


def compute_cosine(input: np.array, df: pd.DataFrame) -> list:
  '''
    Compute recommendation based on one user profile (aka user feature vector) 
    against with all other users in the sample.

    INPUT:
      input: 2D numpy array with 17 user features
      df: pandas dataframe type which contains n user survey data

    OUTPUT:
      result: a list of cosine similarity score
        - note: the order of the list is in the correct corresponding the row of the sample,
                any col shuffle must be extended across the sample.
  '''
  result = list()

  for i in range(len(df.index)):
    result.append(cosine_similarity(
        input,
        np.array([df.iloc[i].values], dtype=np.int32)
        )
    )
  
  return result

  
def product_map(sim_list, customer_data, T):
  # append the list to the dataframe
  customer_data['rank_score'] = sim_list

  # encode the ranking to float datatype
  customer_data['rank_score'] = customer_data['rank_score'].astype(float, errors = 'raise')

  # sort the dataframe by the ranking descending while preserving the ones that above the threshold
  most_similar_user = customer_data[customer_data['rank_score']>T].sort_values('rank_score', ascending=False)

  # select the dish col
  code = most_similar_user['Favorite Dish ']

  return code


def get_recommend(raw_input: list):
  # encode the user feature vector into numpy array with int32 datatype
  input = np.array([raw_input], dtype=np.int32)

  # computer the similarity score listing
  sim_list = compute_cosine(input, df3_drop)
  
  # map the similarity score to the dish code
  product_recommended_code = product_map(sim_list, df3, T)

  # convert the dish code to a list
  items = product_recommended_code.tolist()

  # select the dish name based on the dish code
  return df2.loc[df2['Item ID'].isin(items), 'Item Name (ENG)']
