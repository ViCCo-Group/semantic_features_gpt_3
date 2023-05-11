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

# 317 text concepts
# 10 inits

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

group_to_one_concept = True
duplicates = True 
min_amount_runs_feature_occured_within_concept = 1

# 317 objects are used to calculate similarities
feature_norms = {
    'McRae': load_mcrae(group_to_one_concept, duplicates),
    'CSLB': load_cslb(group_to_one_concept),
    
    'GPT3-McRae-10': load_gpt(1, group_to_one_concept, min_amount_runs_feature_occured_within_concept, duplicates, 'mcrae_priming', 'gpt3-davinci', 10, "test_concepts"),
    'GPT3-CSLB-10': load_gpt(1, group_to_one_concept, min_amount_runs_feature_occured_within_concept, duplicates, 'cslb_priming', 'gpt3-davinci', 10, "test_concepts"),

    'GPT4-McRae-10': load_gpt(1, group_to_one_concept, min_amount_runs_feature_occured_within_concept, duplicates, 'mcrae_priming', 'gpt4', 10, "test_concepts"),

    'Claude-v1-McRae-10': load_gpt(1, group_to_one_concept, min_amount_runs_feature_occured_within_concept, duplicates, 'mcrae_priming', 'claude', 10, "test_concepts")
}

intersection_concepts = generate_concepts_to_keep(feature_norms, 'intersection')
feature_norms_vec, behav_sim_matched = get_all_vectorized(feature_norms, intersection_concepts, 'count')

corr = calc_correlation(feature_norms_vec, behav_sim_matched)
corr.style.background_gradient(cmap='coolwarm', axis=None).set_precision(3)

print(corr)