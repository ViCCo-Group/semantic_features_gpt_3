import sys 
import os 

sys.path.append('../..')

DATA_DIR = '../../data'
os.environ['DATA_DIR'] = DATA_DIR
FIGURES_DIR = '../../figures'

from utils.things import calc_correlation, get_all_vectorized, sort_vec, match_behv_sim
from utils.correlation import calc_semi_partial_correlation
from utils.data import load_gpt, load_cslb, load_mcrae, load_behav
from utils.stats import stats
import matplotlib.pyplot as plt 

from utils.correlation import get_similiarity_vector
from utils.vectorization import vectorize_concepts
from utils.data import load_cslb_count_vec, load_sorting
from utils.feature_norms import generate_concepts_to_keep 

group_to_one_concept = True
duplicates = True 
min_amount_runs_feature_occured = 4
min_amount_runs_feature_occured_within_concept = 1

# 317 objects are used to calculate similarities
feature_norms = {
    'McRae': load_mcrae(group_to_one_concept, duplicates),
    'CSLB': load_cslb(group_to_one_concept),
    'GPT3-davinci-McRae': load_gpt(min_amount_runs_feature_occured, group_to_one_concept, min_amount_runs_feature_occured_within_concept, duplicates, 'mcrae_priming', 'gpt3-davinci', 30, "all_things_concepts"),    
}

# Use concepts that are not used in the final comparison
validation_concepts = generate_concepts_to_keep(feature_norms, 'excl_cslb_mcrae')
behav_sim = load_behav()

### How does the number of runs a featured occured influence the similarity with THINGS?

min_amount_runs_feature_occured = 1
group_to_one_concept = False
min_amount_runs_feature_occured_within_concept = 1
duplicates = True 

gpt_df = load_gpt(min_amount_runs_feature_occured, group_to_one_concept, min_amount_runs_feature_occured_within_concept, duplicates, 'mcrae_priming', 'gpt3-davinci', 30, "all_things_concepts")
behv_sim = match_behv_sim(behav_sim, validation_concepts, load_sorting())

r = []
all_n_features = []
all_n_unique_features = []

for i in range(1, 10):
    gpt_df_temp = gpt_df[gpt_df['amount_runs_feature_occured'] >= i]
    gpt_df_temp = gpt_df_temp[gpt_df_temp['concept_id'].isin(validation_concepts)]

    n_features, n_concepts, n_unique_features, mean_amount_features_per_concept, mean_amount_of_concepts_per_feature, mean_amount_shared_features_per_concept = stats(gpt_df_temp)
    all_n_features.append(n_features)
    all_n_unique_features.append(n_unique_features)

    gpt_df_temp = gpt_df_temp.groupby('concept_id', as_index=False).agg({'feature': lambda x: ';'.join(x)})

    feature_norms = {
        'GPT3-davinci-McRae': gpt_df_temp
    }
    corr = run_things_analyses(feature_norms, validation_concepts)
    r_gpt_behav = corr['THINGS']['GPT3-davinci-McRae']
    r.append(r_gpt_behav)

fig, axes = plt.subplots(2, figsize=(10,10), sharex=True)
axes[1].set_xlabel('Number of runs a feature has to occur')

n = range(1, len(r) + 1)

axes[0].scatter(n, r)
axes[0].set_ylabel('Pearson r with THINGS on valdiation concetps')

axes[1].scatter(n, all_n_unique_features)
axes[1].set_ylabel('Number of unique features')

plt.savefig(f'{FIGURES_DIR}/filter.svg')
