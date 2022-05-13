import argparse
from encode import create_base_question, add_sentence_column
import numpy as np
import pandas as pd 

def write_train_df(df_train, filename):
    df_train = df_train[['concept', 'question', 'answer']]
    df_train.to_csv('output_data/things/%s.csv' % (filename), index=False)
    #dfs = np.split_array(df_train, n_question)
    #for i, df in enumerate(dfs):
    #    df.to_csv('output_data/mcrae/%s_%s.csv' % (filename, i), index=False)

def write_test_df(df_test, filename):
    df_test = df_test[['concept', 'feature']]
    df_test.to_csv('output_data/mcrae/%s.csv' % filename, index=False)

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

def split_train_test(df, n_train_concepts, n_test_concepts):
    df_train = df.iloc[:n_train_concepts]
    df_test = df.iloc[n_train_concepts:n_train_concepts+n_test_concepts]
    return (df_train, df_test)

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

def encode(args):
    df = load_corpus_into_df()
    preprocessed_df = preprocess(df)
    n_train_concepts = int(args.n_train_concepts)
    n_test_concepts = int(args.n_test_concepts)

    output_train_filename = args.output_train
    output_test_filename = args.output_test

    sorted_df = find_words_with_high_overage_of_br_label(preprocessed_df)
    df_train, df_test = split_train_test(sorted_df, n_train_concepts, n_test_concepts)

    # Test concept with correct answers that will be checked against predictions
    if output_test_filename:
        write_test_df(df_test, output_test_filename)

    # Train concepts with correct answers that will be used to initialize the context
    df_train = add_sentence_column(df_train)
    write_train_df(df_train, output_train_filename)

parser = argparse.ArgumentParser()
parser.set_defaults(function=encode)
parser.add_argument("--n_train_concepts", dest='n_train_concepts', default=None)
parser.add_argument("--n_test_concepts", dest='n_test_concepts', default=None)
parser.add_argument("--output_train", dest='output_train')
parser.add_argument("--output_test", dest='output_test', default=None)
