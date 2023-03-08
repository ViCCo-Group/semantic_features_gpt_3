from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
from scipy.spatial.distance import squareform
import math 
import pandas as pd 
from utils.correlation import get_similiarity_vector
from utils.vectorization import vectorize_concepts
from utils.data import load_cslb_count_vec, load_sorting
from utils.feature_norms import generate_concepts_to_keep 

def match_behv_sim(behv_sim, concepts_to_keep, sorting_df):
    concept_positions_to_keep = [sorting_df.index[sorting_df['concept_id'] == concept].tolist()[0] for concept in concepts_to_keep]
    concept_positions_to_keep = sorted(concept_positions_to_keep)
    behv_sim_matched = behv_sim[concept_positions_to_keep, :]
    behv_sim_matched = behv_sim_matched[:, concept_positions_to_keep]
    return behv_sim_matched

def calc_correlation(feature_norms_vec, things_similarity, method='pearson'):
    values = {
        "THINGS": squareform(things_similarity, force='tovector', checks=False)
    }    

    for name, feature_norm_vec in feature_norms_vec.items():
        print(name)
        print(feature_norm_vec.shape)
        similarity_vector = get_similiarity_vector(feature_norm_vec)
        print(len(similarity_vector))
        values[name] = similarity_vector

    return pd.DataFrame(values).corr(method)

def sort_vec(df):
    sorted_df = load_sorting().reset_index().set_index('concept_id')
    df['concept_num'] = df.index.map(sorted_df['index'])
    df = df.sort_values(by='concept_num')
    df = df.drop('concept_num', axis=1)
    return df


def get_all_vectorized(feature_norms, behv_sim, vec = 'binary'):
    feature_norms_vec = {}
    sorting = load_sorting()

    intersection_concepts = generate_concepts_to_keep(feature_norms)
    behv_sim = match_behv_sim(behv_sim, intersection_concepts, load_sorting())

    for name, feature_norm in feature_norms.items():
        feature_norm_vec = vectorize_concepts(feature_norm, sorting, vec)

        if vec == 'count' and name == 'CSLB':
            feature_norm_vec = load_cslb_count_vec()
        
        feature_norm_vec = feature_norm_vec.loc[intersection_concepts]
        feature_norms_vec[name] = sort_vec(feature_norm_vec)
        
    return feature_norms_vec, behv_sim