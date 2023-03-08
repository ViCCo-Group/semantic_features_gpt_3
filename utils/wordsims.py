from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import pandas as pd 
from copy import deepcopy
from utils.vectorization import vectorize_concepts
from utils.data import load_cslb_count_vec

def calc_correlation(feature_norms_similarity_ratings, wordsim_similarity_ratings, wordsim_dataset_name, method='pearson'):
    all_ratings = deepcopy(feature_norms_similarity_ratings)
    all_ratings[wordsim_dataset_name] = wordsim_similarity_ratings
    return pd.DataFrame(all_ratings).corr(method)

def calc_pair_sim(df_vec, concept1, concept2, ):
    df1 = pd.DataFrame(df_vec.loc[concept1]).transpose()
    df2 = pd.DataFrame(df_vec.loc[concept2]).transpose()
    sim = cosine_similarity(df1, df2)[0][0]
    return sim

def get_all_vectorized(feature_norms, vec = 'binary'):
    feature_norms_vec = {}

    for name, feature_norm in feature_norms.items():
        feature_norm_vec = vectorize_concepts(feature_norm, None, vec)

        if vec == 'count' and name == 'cslb':
            feature_norm_vec = load_cslb_count_vec()
        
        feature_norms_vec[name] = feature_norm_vec
        
    return feature_norms_vec