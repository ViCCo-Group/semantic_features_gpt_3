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
        "axes.labelsize": 5,
        "axes.titlesize": 5,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 5,
        "xtick.labelsize": 3,
        "ytick.labelsize": 3
}
plt.rcParams.update(tex_fonts)

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

_, _, animal = get_categories(intersection_concepts)

for feature_norm_name, feature_norm_vec in feature_norms_vec.items():
    vec_animal = feature_norm_vec.loc[animal]
    similarities = cosine_similarity(vec_animal, vec_animal)
    similarities = pd.DataFrame(similarities, columns=animal, index=animal)

    a = sns.clustermap(similarities, yticklabels=True, xticklabels=True, figsize=(3,3))
    #a.fig.suptitle(f'{feature_norm_name}', y=1) 
    plt.tight_layout()
    plt.savefig(pjoin(FIGURES_DIR, f'animals-{feature_norm_name}.svg'))