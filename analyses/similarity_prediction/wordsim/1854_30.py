import sys 
import os 

sys.path.append('../../..')

DATA_DIR = '../../../data'
os.environ['DATA_DIR'] = DATA_DIR

from utils.wordsims import calc_correlation, calc_pair_sim, get_all_vectorized
from utils.data import load_gpt, load_cslb, load_behav, load_mcrae
import numpy as np
import pandas as pd
from collections import defaultdict
from IPython.display import display
from os.path import join as pjoin

# Predicting human similarity judgements
## Wrd Similarity and relatedness

group_to_one_concept = True
duplicates = True 
min_amount_runs_feature_occured = 4
min_amount_runs_feature_occured_within_concept = 1
number_inits = 30

feature_norms = {
    'McRae': load_mcrae(group_to_one_concept, duplicates),
    'CSLB': load_cslb(group_to_one_concept),
    'GPT3-McRae-30': load_gpt(min_amount_runs_feature_occured, group_to_one_concept, min_amount_runs_feature_occured_within_concept, duplicates, 'mcrae_priming', 'gpt3-davinci', number_inits, "all_things_concepts"),
    'GPT3-CSLB-30': load_gpt(min_amount_runs_feature_occured, group_to_one_concept, min_amount_runs_feature_occured_within_concept, duplicates, 'cslb_priming', 'gpt3-davinci', number_inits, "all_things_concepts"),
    'ChatGPT3-McRae-30': load_gpt(min_amount_runs_feature_occured, group_to_one_concept, min_amount_runs_feature_occured_within_concept, duplicates, 'mcrae_priming', 'chatgpt-gpt3.5-turbo', number_inits, "all_things_concepts")
}

feature_norms_vec = get_all_vectorized(feature_norms, 'count')

wordsim = pd.read_csv(pjoin(DATA_DIR, 'wordsim353/combined.csv'))
wordsim = wordsim.rename(columns={'Word 1': 'word1', 'Word 2': 'word2', 'Human (mean)': 'rating'})

simlex = pd.read_csv(pjoin(DATA_DIR, 'simlex999/SimLex-999.txt'), sep='\t', usecols=['word1', 'word2', 'SimLex999'])
simlex = simlex.rename(columns={'SimLex999': 'rating'})

men = pd.read_csv(pjoin(DATA_DIR, 'men/MEN_dataset_natural_form_full'), sep=' ', names=['word1', 'word2', 'rating'], header=None)

yp = pd.read_csv(pjoin(DATA_DIR, 'yp/yp-130.csv'))
yp = yp.rename(columns={'similarity': 'rating'})

mturk771 = pd.read_csv(pjoin(DATA_DIR, 'mturk_771/mturk-771.csv'))
mturk771 = mturk771.rename(columns={'similarity': 'rating'})

mturk287 = pd.read_csv(pjoin(DATA_DIR, 'mturk_287/mturk-287.csv'))
mturk287 = mturk287.rename(columns={'similarity': 'rating'})

rw = pd.read_csv(pjoin(DATA_DIR, 'rw/rw.csv'))
rw = rw.rename(columns={'similarity': 'rating'})


datasets = (('wordsim-353', wordsim), 
            ('simlex-999', simlex), 
            ('men', men),
            ('mturk-771', mturk771),
            ('mturk-287', mturk287),
            ('rw', rw),
            ('yp', yp))

for dataset_name, wordsim_df in (('simlex-999', simlex),('men', men)):
    print(dataset_name)
    wordsim_ratings = []
    
    feature_norms_similarity_ratings = defaultdict(lambda: [])
    
    for row in wordsim_df.iterrows():
        word1 = row[1]['word1'].lower()
        word2 = row[1]['word2'].lower()
        rating = row[1]['rating']
        
        # change homonym word to the THINGS ID that is used in all feature norms
        gpt_words = feature_norms_vec['GPT3-McRae-30'].index
        if word1 not in gpt_words:
            word1 = f'{word1}1'
            
        if word2 not in gpt_words:
            word2 = f'{word2}1'
        
       
        # check if both words exists in all feature norms, if not skip
        words_exists_in_all_norms = True
        for norm_name, feature_vec in feature_norms_vec.items():
            word1_exists_in_norm = word1 in feature_vec.index
            word2_exists_in_norm = word2 in feature_vec.index
            if not word1_exists_in_norm or not word2_exists_in_norm:
                words_exists_in_all_norms = False 
                break 
        if not words_exists_in_all_norms:
            continue
        
        wordsim_ratings.append(rating)

        for norm_name, feature_vec in feature_norms_vec.items():
            similarity = calc_pair_sim(feature_vec, word1, word2)
            feature_norms_similarity_ratings[norm_name].append(similarity)

    corr_df = calc_correlation(feature_norms_similarity_ratings, wordsim_ratings, dataset_name)
    print(f'{len(wordsim_ratings)} word pairs')
    print(corr_df)
        