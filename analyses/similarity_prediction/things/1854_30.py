import sys 
import os 

sys.path.append('../../..')

DATA_DIR = '../../../data'
os.environ['DATA_DIR'] = DATA_DIR

from utils.things import calc_correlation, sort_vec, match_behv_sim
from utils.correlation import calc_semi_partial_correlation
from utils.data import load_gpt, load_cslb, load_mcrae, load_behav
import matplotlib.pyplot as plt 

from utils.correlation import get_similiarity_vector
from utils.vectorization import vectorize_concepts
from utils.data import load_cslb_count_vec, load_sorting
from utils.feature_norms import generate_concepts_to_keep 
from utils.concepts import load_test_concepts, load_all_things_concepts, load_val_concepts

#RSA with SpoSe embeddings
#Use feature norms that were generated with 30 inits and for 1854 concepts
#Use 317 THINGS objects in the similarity matrix that are present in all feature norms

group_to_one_concept = True
duplicates = True 
min_amount_runs_feature_occured = 1
min_amount_runs_feature_occured_within_concept = 1
number_inits = 30

# 317 objects are used to calculate similarities
feature_norms = {
    'McRae': load_mcrae(group_to_one_concept, duplicates),
    'CSLB': load_cslb(group_to_one_concept),
    'GPT3-McRae': load_gpt(min_amount_runs_feature_occured, group_to_one_concept, min_amount_runs_feature_occured_within_concept, duplicates, 'mcrae_priming', 'gpt3-davinci', number_inits, "all_things_concepts"),
    'GPT3-CSLB': load_gpt(min_amount_runs_feature_occured, group_to_one_concept, min_amount_runs_feature_occured_within_concept, duplicates, 'cslb_priming', 'gpt3-davinci', number_inits, "all_things_concepts"),
    'ChatGPT3-McRae': load_gpt(min_amount_runs_feature_occured, group_to_one_concept, min_amount_runs_feature_occured_within_concept, duplicates, 'mcrae_priming', 'chatgpt-gpt3.5-turbo', number_inits, "all_things_concepts"),
    'Claude-v1-McRae': load_gpt(min_amount_runs_feature_occured, group_to_one_concept, min_amount_runs_feature_occured_within_concept, duplicates, 'mcrae_priming', 'claude', 30, "all_things_concepts")
}

intersection_concepts = generate_concepts_to_keep(feature_norms, 'intersection')
run_things_analyses(feature_norms, intersection_concepts)