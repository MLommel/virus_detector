# import std libraries
import numpy as np
import pandas as pd
import time

from IPython.display import HTML
import pickle
import json

#import models
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import NMF 
from sklearn.neighbors import KNeighborsClassifier
from scipy.sparse import csr_matrix

import streamlit as st

###################################################
#FOR MODELS

# Data analysis stack
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning stack
from sklearn.model_selection import (
    train_test_split,
    RepeatedStratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV,
    KFold
)
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
    FunctionTransformer,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsClassifier

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    log_loss,
    mean_absolute_error
)
from sklearn.utils.validation import check_is_fitted
from scipy.stats import randint, loguniform

# Miscellaneous
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

#############################################
#pip install streamlit-aggrid
#pip install streamlit
dfN = pd.read_csv('df.csv', index_col=0)
XtrainN = pd.read_csv('Xtrain.csv', index_col=0)
#XtestN = pd.read_csv('Xtest.csv', index_col=0)
ytrain = pd.read_csv('ytrain.csv',dtype=int, index_col=0)
#ytest = pd.read_csv('ytest.csv', index_col=0)
df_trainN = XtrainN.merge(ytrain, left_index=True, right_index=True, how='left')
def recommend_nn(query, model,Rt, k=10):
    """
    Filters and recommends the top k movies for any given input query based on a trained NMF model. 
    Returns a list of k movie ids.
    """
        
    # 1. construct new_user-item dataframe given the query
    new_user_dataframe =  pd.DataFrame(query, columns=movies['title'], index=['new_user'])
    #print(new_user_dataframe)
    # 1.2. fill the NaN
    new_user_dataframe_imputed = new_user_dataframe.fillna(0) #better mean
    # 2. scoring
    # calculates the distances to all other users in the data!
    similarity_scores, neighbor_ids = model.kneighbors(
    new_user_dataframe_imputed,
    n_neighbors=15,
    return_distance=True
    )

    # sklearn returns a list of predictions
    # extract the first and only value of the list

    neighbors_df = pd.DataFrame(
    data = {'neighbor_id': neighbor_ids[0], 'similarity_score': similarity_scores[0]}
    )
    
    # 3. ranking
    # only look at ratings for users that are similar!
    neighborhood = Rt.iloc[neighbor_ids[0]]
  
    
        # filter out movies already seen by the user
    neighborhood_filtered = neighborhood.drop(query.keys(),axis=1)
   

    # calculate the summed up rating for each movie
    # summing up introduces a bias for popular movies
    # averaging introduces bias for movies only seen by few users in the neighboorhood

    df_score = neighborhood_filtered.sum().sort_values(ascending=False)
    
    # return the top-k highest rated movie ids or titles
    df_score_ranked = df_score.sort_values(ascending=False).index.tolist()
    recommended = df_score_ranked[:k]
    return recommended#, df_score
def recommend_nmf(query, model, k=10):
    """
    Filters and recommends the top k movies for any given input query based on a trained NMF model. 
    Returns a list of k movie ids.
    """
    
    # 1. construct new_user-item dataframe given the query(votings of the new user)
   
    new_user_dataframe = pd.DataFrame(query, columns=movies['title'], index=['new_user'])
   
    new_user_dataframe_imputed =new_user_dataframe.fillna(0)

    P_new_user_matrix = model.transform(new_user_dataframe_imputed)
    # get as dataframe for a better visualizarion
    P_new_user = pd.DataFrame(P_new_user_matrix, 
                         columns = model.get_feature_names_out(),
                         index = ['new_user'])
    
    Q_matrix = model.components_
    Q = pd.DataFrame(Q_matrix, columns=movies['title'], index=model.get_feature_names_out())

    R_hat_new_user_matrix = np.dot(P_new_user,Q)
    # get as dataframe for a better visualizarion
    R_hat_new_user = pd.DataFrame(data=R_hat_new_user_matrix,
                         columns=movies['title'],
                         index = ['new_user'])
    R_hat_new_filtered = R_hat_new_user#.drop(new_user_query.keys(), axis=1)
    R_hat_new_filtered.T.sort_values(by=["new_user"], ascending=False).index.tolist()
    ranked =  R_hat_new_filtered.T.sort_values(by=["new_user"], ascending=False).index.tolist()
    recommended = ranked[:k]
    return recommended#, R_hat_new_filtered.T.sort_values(by=["new_user"], ascending=False)
BEST_MOVIES = df_trainN #pd.read_csv("best_movies.csv")
#BEST_MOVIES.rename(
  #  index=lambda x: x+1,
   # inplace=True
   # )
TITLES = ["---"] + list(BEST_MOVIES['p1'].sort_values()) 
with open('model_rdmf.pkl', 'rb') as file:
    DISTANCE_MODEL = pickle.load(file)

with open('model_rdmf.pkl', 'rb') as file:
    NMF_MODEL = pickle.load(file)
 
'''

new_user_query = {"Toy Story (1995)":5,
                 "Grumpier Old Men (1995)":2,
                 "Casino (1995)":3.5,
                 "Sabrina (1995)":4,
                 "GoldenEye (1995)":5}
print(new_user_query)


#AgGrid(BEST_MOVIES.head(20))
print("end")
#print([movies])
new_user_dataframe = pd.DataFrame(new_user_query, columns=movies['title'], index=[0])
new_user_dataframe_imputed =new_user_dataframe.fillna(0)
#type(new_user_dataframe)
#print(BEST_MOVIES)
#Ru = pd.DataFrame(data=new_user_dataframe_imputed, columns=movies['title'],index = UserId)
print(new_user_dataframe_imputed)
similarity_scores, neighbor_ids = DISTANCE_MODEL.kneighbors(
    new_user_dataframe_imputed,
    n_neighbors=15,
    return_distance=True
    )
'''
#recommend_nn(new_user_dataframe_imputed, DISTANCE_MODEL,Rt, k=10)
#pip install pipreqs