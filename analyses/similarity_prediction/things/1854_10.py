import sys 
import os 

sys.path.append('../..')

DATA_DIR = '../../data'
os.environ['DATA_DIR'] = DATA_DIR

from utils.things import calc_correlation, get_all_vectorized, sort_vec, match_behv_sim
from utils.correlation import calc_semi_partial_correlation
from utils.data import load_gpt, load_cslb, load_mcrae, load_behav
import matplotlib.pyplot as plt 

from utils.correlation import get_similiarity_vector
from utils.vectorization import vectorize_concepts
from utils.data import load_cslb_count_vec, load_sorting
from utils.feature_norms import generate_concepts_to_keep 

# RSA with SpoSe embeddings
# Use feature norms that were generated with 10 inits and for 1854 concepts
# Use all 1854 THINGS objects in the similarity matrix

group_to_one_concept = True
duplicates = True 
min_amount_runs_feature_occured_within_concept = 1

feature_norms = {
    'GPT3-McRae-10': load_gpt(1, group_to_one_concept, min_amount_runs_feature_occured_within_concept, duplicates, 'mcrae_priming', 'gpt3-davinci', 10, "all_things_concepts"),
    'GPT3-CSLB-10': load_gpt(1, group_to_one_concept, min_amount_runs_feature_occured_within_concept, duplicates, 'cslb_priming', 'gpt3-davinci', 10, "all_things_concepts"),
}

all_things_concepts = list(feature_norms['GPT3-CSLB-10']['concept_id'].unique())
run_things_analyses(feature_norms, all_things_concepts)