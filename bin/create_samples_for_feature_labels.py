import sys
import os 

sys.path.append('..')

DATA_DIR = '../data'
os.environ['DATA_DIR'] = DATA_DIR

import pandas as pd 
from utils.feature_norms import generate_concepts_to_keep
from utils.feature_labels import get_feature_label
import argparse

def run(args):    
    n_intersection = 500
    n_diff = 500

    feature_norm = pd.read_csv(args.path_to_feature_norm)
    intersection_concepts = []
    with open(os.path.join(DATA_DIR, 'concepts', 'test_concepts.txt')) as concepts_file:
        for line in concepts_file:
            intersection_concepts.append(line.rsplit()[0])
    feature_norm_intersection_concepts = pd.DataFrame({'feature': feature_norm[feature_norm['concept_id'].isin(intersection_concepts)]['decoded_feature'].unique()}).sample(n_intersection)

    unique_concepts = set(feature_norm['concept_id']).difference(intersection_concepts)
    feature_norm_unique_concepts = pd.DataFrame({'feature': feature_norm[feature_norm['concept_id'].isin(unique_concepts)]['decoded_feature'].unique()}).sample(n_diff)

    feature_norm_sample = pd.concat([feature_norm_intersection_concepts, feature_norm_unique_concepts])
    feature_norm_sample['label'] = feature_norm_sample['feature'].apply(get_feature_label)
    feature_norm_sample.to_csv('sample.csv', index=False)

parser = argparse.ArgumentParser()
parser.set_defaults(function=run)
parser.add_argument("--path_to_feature_norm", dest='path_to_feature_norm')

args = parser.parse_args()
args.function(args)