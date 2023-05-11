import sys 
import os 

sys.path.append('../..')

DATA_DIR = '../../data'
os.environ['DATA_DIR'] = DATA_DIR

FIGURES_DIR = '../../figures'

from utils.things import calc_correlation
from utils.data import load_things, load_gpt, load_cslb, load_sorting, load_mcrae, load_behav
from utils.things import match_behv_sim
from utils.feature_norms import generate_concepts_to_keep
from utils.analyses.category.category import get_categories
from utils.analyses.category.pairiwise import calc_sim, plot_violin
from utils.things import get_all_vectorized
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from os.path import join as pjoin

group_to_one_concept = True
duplicates = True 
min_amount_runs_feature_occured = 4
min_amount_runs_feature_occured_within_concept = 1

feature_norms = {
    'McRae': load_mcrae(group_to_one_concept, duplicates),
    'CSLB': load_cslb(group_to_one_concept),
    'GPT3-davinci-McRae': load_gpt(min_amount_runs_feature_occured, group_to_one_concept, min_amount_runs_feature_occured_within_concept, duplicates, "mcrae_priming", "gpt3-davinci", 30, "all_things_concepts"),
    }

intersection_concepts = generate_concepts_to_keep(feature_norms, 'intersection')
feature_norms_vec, behav_sim_matched = get_all_vectorized(feature_norms, intersection_concepts, 'count')
categories, _, _ = get_categories(intersection_concepts)
sims_gpt, sims_cslb, sims_mc = calc_sim(feature_norms_vec['GPT3-davinci-McRae'], feature_norms_vec['CSLB'], feature_norms_vec['McRae'], categories)

# TODO count/tfidf not wroking csbl mcrae

fig, axes = plt.subplots(1,1, figsize=(33,5), sharex=True)
plot_violin(axes, sims_gpt, sims_cslb, sims_mc, categories)
plt.tight_layout()
plt.savefig(pjoin(FIGURES_DIR, 'pairwise_similarities.svg'))
