import pandas as pd 
from data import generate_concepts_to_keep, load_gpt, load_cslb, load_mcrae
from utils.feature_labels import get_feature_label
import argparse

def run(args):
    output_dir = args.output_dir
    
    n_intersection = 500
    n_diff = 500

    gpt_df = load_gpt(1,False,1,1,False)
    clsb_df = load_cslb(False)
    mc_df = load_mcrae(False, False)

    feature_norm = pd.read_csv(args.path_to_feature_norm)

    intersection_concepts = generate_concepts_to_keep(gpt_df, mc_df, clsb_df, None, 'intersection')
    feature_norm_intersection_concepts = pd.DataFrame({'feature': feature_norm[feature_norm['concept_id'].isin(intersection_concepts)]['feature'].unique()}).sample(n_intersection)

    #diff_concepts_mc = generate_concepts_to_keep(gpt_df, mc_df, clsb_df, None, 'exclusive_mc_concepts')
    unique_concepts = set(feature_norm['concept_id']).difference(intersection_concepts)
    feature_norm_unique_concepts = pd.DataFrame({'feature': feature_norm[feature_norm['concept_id'].isin(unique_concepts)]['feature'].unique()}).sample(n_diff)

    feature_norm_sample = pd.concat([feature_norm_intersection_concepts, feature_norm_unique_concepts])
    feature_norm_sample['label'] = feature_norm_sample['feature'].apply(get_feature_label)
    feature_norm_sample.to_csv(f'{output_dir}.csv', index=False)

parser = argparse.ArgumentParser()
parser.set_defaults(function=run)
parser.add_argument("--output_dir", dest='output_dir')
parser.add_argument("--path_to_feature_norm", dest='path_to_feature_norm')

args = parser.parse_args()
args.function(args)