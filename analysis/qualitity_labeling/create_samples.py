from dataclasses import replace
import sys 
import os 

sys.path.append('../..')

DATA_DIR = '../../data'
os.environ['DATA_DIR'] = DATA_DIR

from copy import deepcopy
from utils.data import generate_concepts_to_keep, load_cslb, load_gpt, load_mcrae, match_gpt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


def load_gpt_with_overlap():
    gpt_df = load_gpt(4, False, 0, None, False)
    mc_df = load_mcrae(False, False)
    cslb_df = load_cslb(False)

    overlap = generate_concepts_to_keep(gpt_df, mc_df, cslb_df, None)
    return match_gpt(gpt_df, overlap)

combs = [
    {
        'name': 'mcrae',
        'df': load_mcrae(False, False),
        'source': 'human'
    },
    {
        'name': 'cslb',
        'df': load_cslb(False, True),
        'source': 'human'
    },
    {
        'name': 'gpt_3_filtered',
        'df': load_gpt(4, False, 0, None, False),
        'source': 'gpt'
    },
    {
        'name': 'gpt_3_unfiltered',
        'df': load_gpt(0, False, 0, None, False),
        'source': 'gpt'
    }
    ,
    {
        'name': 'gpt_3_filtered_overlap',
        'df': load_gpt_with_overlap(),
        'source': 'gpt'
    }
]

all_samples_person1 = pd.DataFrame()
all_samples_person2 = pd.DataFrame()

for comb in combs:
    name = comb['name']
    df = comb['df']
    source = comb['source']

    df = df[['concept_id', 'feature']]
    df['name'] = name
    df['source'] = source 

    # Features that are shared between labeller 
    n_samples = 100
    sample = df.sample(n_samples, replace=False)
    all_samples_person1 = pd.concat([all_samples_person1, sample])
    all_samples_person2 = pd.concat([all_samples_person2, sample])

    # Features that are unique for the labeller
    n_samples = 300

    sample1 = df.sample(n_samples, replace=False)
    all_samples_person1 = pd.concat([all_samples_person1, sample1])

    sample2 = df.sample(n_samples, replace=False)
    all_samples_person2 = pd.concat([all_samples_person2, sample2])

for i, samples_df in enumerate([all_samples_person1, all_samples_person2]):
    # shuffle all rows 
    samples_df = samples_df.sample(frac=1)

    # generate ID
    samples_df = samples_df.reset_index(drop=True).reset_index()

    # truth
    samples_df.to_csv(f'sample_with_true_source_{i}.csv', index=False)

    # CSV to be labelled 
    samples_df[['index', 'concept_id', 'feature']].to_csv(f'sample_{i}.csv', index=False)
