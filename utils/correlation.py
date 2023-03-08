from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
from scipy.spatial.distance import squareform
import math 
import pandas as pd 

def get_similiarity_vector(df_vec, name=""):
    # Build cosine similarity matrices of features and flatten triangular part
    #data_dir = './evaluation/check_data'
    pred_sim_matrix = cosine_similarity(df_vec, df_vec)
    #pd.DataFrame(pred_sim_matrix).to_csv('%s/similarity_matrix_%s.csv' % (data_dir, name))
    sim = squareform(pred_sim_matrix, force='tovector', checks=False)
    return sim

def calc_semi_partial_correlation(r_12, r_13, r_23):
    semipartial_23 = (r_12 - (r_13*r_23)) / (math.sqrt(1-(r_23 ** 2)))
    return semipartial_23 




