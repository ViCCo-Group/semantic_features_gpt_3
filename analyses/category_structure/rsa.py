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


# Predicting superordinate categories

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
intersection_concepts = generate_concepts_to_keep(feature_norms)

#things_df_count = things_df.groupby('category', as_index=False).count().sort_values(by='concept_id', ascending=False)

_, all_short_categories, _ = get_categories(intersection_concepts)

for feature_norm_name, feature_norm_vec in feature_norms_vec.items():
    vec_intersection = feature_norm_vec.loc[all_short_categories]
    similarities = cosine_similarity(vec_intersection, vec_intersection)
    similarities = pd.DataFrame(similarities, columns=all_short_categories, index=all_short_categories)

    fig, ax = plt.subplots(figsize=(15,15))
    sns.heatmap(similarities, ax=ax, yticklabels=True, xticklabels=True, cbar=False)
    plt.tight_layout()
    plt.title(feature_norm_name)
    plt.savefig(pjoin(FIGURES_DIR, f'all-catergories-{feature_norm_name}.svg'))

