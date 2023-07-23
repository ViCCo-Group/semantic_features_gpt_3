import argparse
from encode import add_sentence_column
import numpy as np
import pandas as pd 

def write_train_df(df_train, filename):
    df_train = df_train[['concept', 'question', 'answer']]
    df_train.to_csv('output_data/things/%s.csv' % (filename), index=False)

def load_corpus_into_df():
    return pd.read_excel('input_data/mcrae/CONCS_FEATS_concstats_brm.xlsx')

def find_words_with_high_overage_of_br_label(df):
    number_unique_wb_labels = len(df['WB_Label'].unique())
    number_unique_br_labels = len(df['BR_Label'].unique())

    def wb_count(labels):
        return len(set(labels))/number_unique_wb_labels * 100

    def br_count(labels):
        return len(set(labels))/number_unique_br_labels * 100

    def join_to_string(values):
        return ','.join(set(values))

    df['br_count'] = df['BR_Label'].groupby(df['concept']).transform(br_count)
    df['wb_count'] = df['WB_Label'].groupby(df['concept']).transform(wb_count)
    df_groubed = df.groupby('concept', as_index=False).agg({'wb_count': 'first', 'br_count': 'first', 'BR_Label': join_to_string, 'WB_Label': join_to_string, 'feature': join_to_string})
    df_groubed = df_groubed.sort_values(['br_count', 'wb_count'], ascending=False)
    df_groubed.to_csv('processed_input.csv')
    return df_groubed
    
def preprocess(df):
    df = df[df['WB_Label'] != 'subordinate']
    df = df.rename({'Concept': 'concept', 'Feature': 'feature'}, axis=1)
    def remove_beh(feature):
        if feature.startswith('beh') or feature.startswith('inbeh'):
            feature_split = feature.split('_')
            clean_feature_split = feature_split[2:]
            return '_'.join(clean_feature_split)
        return feature
    df['feature'] = df['feature'].apply(remove_beh)
    return df 

def load_corpus_into_df():
    return pd.read_excel('input_data/mcrae/CONCS_FEATS_concstats_brm.xlsx')

def encode():
    df = load_corpus_into_df()
    preprocessed_df = preprocess(df)

    # Train concepts with correct answers that will be used to initialize the context
    df_train = add_sentence_column(df_train, 20, True, '_')
    write_train_df(df_train, output_train_filename)

