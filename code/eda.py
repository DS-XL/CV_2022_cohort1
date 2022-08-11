import pandas as pd
import numpy as np
import os
from pandas_profiling import ProfileReport

F1 = "./datasets/MeetFresh_Menu_Drink.csv"
F2 = "./datasets/MeetFresh_Menu_Food.csv"
F3 = "./datasets/survey.csv"

# define the threshold for the cosine similarity
T = 0.9980

df1 = pd.read_csv(F1)
df2 = pd.read_csv(F2)
df3 = pd.read_csv(F3)

profile_df1 = ProfileReport(df1, title="MeetFresh_Menu_Drink")
profile_df2 = ProfileReport(df2, title="MeetFresh_Menu_Food")
profile_df3 = ProfileReport(df3, title="MeetFresh_survey")

profile_df1.to_file("MeetFresh_Menu_Drink.html")
profile_df2.to_file("MeetFresh_Menu_Food.html")
profile_df3.to_file("MeetFresh_survey.html")