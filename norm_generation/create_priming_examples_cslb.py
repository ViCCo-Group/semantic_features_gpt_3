import sys 
sys.path.append('../..')
import os 
DATA_DIR = '../../data'
os.environ['DATA_DIR'] = DATA_DIR

from encode import add_sentence_column
import inflect as word
inflect = word.engine()

import pandas as pd 
import numpy as np
import random

def load_cslb():
    return pd.read_csv('data/cslb/norms.dat', sep='\t')[['concept', 'feature']]

def write_train_trial_dfs(df, n_trials):
    df = df[['concept', 'question', 'answer']]
    trial_dfs = np.array_split(df, n_trials)
    for i, trial_df in enumerate(trial_dfs):
        trial_df.to_csv(f'data/priming_examples/cslb/train_{str(i+1)}.csv', index=False)

def generate_random_concepts(df, n_concepts):
    cslb_concepts = set(df['concept'])
    print(len(cslb_concepts))
    cslb_concepts = cslb_concepts.difference(load_things_concepts())
    print(cslb_concepts)

    concepts = random.sample(cslb_concepts, n_concepts)
    return concepts

def create_sub_df(df, concepts):
    df = df[df['concept'].isin(concepts)]
    df = df.groupby('concept', as_index=False).agg({'feature': lambda features: ','.join(features)})
    return df.sample(frac=1)

def preprocess(df):
    def transform_concept(concept):
        concept = concept.replace('_', ' ')     
        return concept

    df['concept'] = df['concept'].apply(transform_concept)
    df['feature'] = df['feature'].apply(lambda feature: feature.split('_')[0])
    return df 

def encode():
    cslb_df = load_cslb()
    df = preprocess(cslb_df)

    n_concepts_per_trial = 3
    n_trials = 30
    n_concepts = n_concepts_per_trial * n_trials
    max_n_features_per_concept = 13

    concepts = generate_random_concepts(df, n_concepts)
    df = create_sub_df(cslb_df, concepts)

    df = add_sentence_column(df, max_n_features_per_concept, check_plural=True, feature_seperator=' ')

    write_train_trial_dfs(df, n_trials)

