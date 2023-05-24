import sys 
import os 

sys.path.append('../..')

DATA_DIR = '../../data'
os.environ['DATA_DIR'] = DATA_DIR

FIGURES_DIR = '../../figures'

from utils.data import load_things, load_gpt, load_cslb, load_sorting, load_mcrae, load_behav
from utils.things import match_behv_sim
from utils.feature_norms import generate_concepts_to_keep
from utils.analyses.category.category import get_categories
from utils.analyses.category.pairiwise import calc_sim, plot_violin
from utils.things import vectorize_filter_sort_feature_norms
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from os.path import join as pjoin

tex_fonts = {
        # Use LaTeX to write all text
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 5,
        "ytick.labelsize": 5
}
plt.rcParams.update(tex_fonts)

# Predicting superordinate categories

group_to_one_concept = True
duplicates = True 
min_amount_runs_feature_occured = 4
min_amount_runs_feature_occured_within_concept = 1

feature_norms = {
    'McRae': load_mcrae(group_to_one_concept, duplicates),
    'CSLB': load_cslb(group_to_one_concept),
    'GPT3-davinci-McRae': load_gpt(min_amount_runs_feature_occured, group_to_one_concept, min_amount_runs_feature_occured_within_concept, duplicates, "mcrae_priming", "gpt3-davinci", 30, "all_things_concepts"),
    'GPT3-davinci-CSLB': load_gpt(min_amount_runs_feature_occured, group_to_one_concept, min_amount_runs_feature_occured_within_concept, duplicates, "cslb_priming", "gpt3-davinci", 30, "all_things_concepts"),
}

intersection_concepts = generate_concepts_to_keep(feature_norms, 'intersection')
feature_norms_vec = vectorize_filter_sort_feature_norms(feature_norms, intersection_concepts, 'count')
intersection_concepts = generate_concepts_to_keep(feature_norms)

#things_df_count = things_df.groupby('category', as_index=False).count().sort_values(by='concept_id', ascending=False)

_, all_short_categories, _ = get_categories(intersection_concepts)

similarities = {}

for feature_norm_name, feature_norm_vec in feature_norms_vec.items():
    vec_intersection = feature_norm_vec.loc[all_short_categories]
    sim = cosine_similarity(vec_intersection, vec_intersection)
    sim = pd.DataFrame(sim, columns=all_short_categories, index=all_short_categories)
    similarities[feature_norm_name] = sim

fig, ax = plt.subplots(2,2, figsize=(7, 7))

sns.heatmap(similarities['McRae'], ax=ax[0][0], yticklabels=True, xticklabels=True, cbar=False)
ax[0][0].set_title('McRae')

sns.heatmap(similarities['CSLB'], ax=ax[1][0], yticklabels=True, xticklabels=True, cbar=False)
ax[1][0].set_title('CSLB')

sns.heatmap(similarities['GPT3-davinci-McRae'], ax=ax[0][1], yticklabels=True, xticklabels=True, cbar=False)
ax[0][1].set_title('GPT3-davinci-McRae')

sns.heatmap(similarities['GPT3-davinci-CSLB'], ax=ax[1][1], yticklabels=True, xticklabels=True, cbar=False)
ax[1][1].set_title('GPT3-davinci-CSLB')

plt.tight_layout()
plt.savefig(pjoin(FIGURES_DIR, f'short-categories.svg'))

