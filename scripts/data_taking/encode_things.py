import argparse
from encode import create_base_question
import pandas as pd  
import numpy as np 

def load_corpus_into_df():
    return pd.read_csv('input_data/things/Wordlist_ratings-Final.csv')

def write_df(df, output_dir):
    df = df[['concept', 'question', 'category', 'id']]
    df.to_csv('%s/concepts.csv' % (output_dir), index=False)

def sort_by_uniqueness(df):
    df['duplicate'] = df['concept'].duplicated(keep=False)
    df_unique = df[df['duplicate'] == False]
    df_not_unqiue = df[df['duplicate'] == True]
    df = pd.concat([df_unique, df_not_unqiue])
    return df

def create_question(row, words_without_duplicates_but_multiple_meanings):
    is_singular = True
    category = False
    if row.plural == 'plural':
        is_singular = False

    concept_is_in_multiple_meaning_list = row.id in words_without_duplicates_but_multiple_meanings['ID'].unique()
    if row.duplicate or concept_is_in_multiple_meaning_list:
        category = row.category
    
    # if things category is overwritten in multiple meanings CSV
    if concept_is_in_multiple_meaning_list:
        overwritten_category = words_without_duplicates_but_multiple_meanings[words_without_duplicates_but_multiple_meanings.ID == row.id]['category'].unique()[0]
        if not pd.isna(overwritten_category):
            category = overwritten_category


    return create_base_question(row.concept, check_plural=False, is_singular=is_singular, category=category)

def preprocess_things(df):
    df = df.rename({'Singular/Plural': 'plural', 'Word': 'concept', 'All Bottom-up Categories': 'category', 'uniqueID': 'id'}, axis=1)
    return df

def load_multiple_words():
    multiple_df = pd.read_csv('input_data/things/words_with_multiple_meaning.csv', sep=';')
    return multiple_df

def encode(args):
    # Things concepts
    df = load_corpus_into_df()
    preprocessed_df = preprocess_things(df)
    words_without_duplicates_but_multiple_meanings = load_multiple_words()
    sorted_df = sort_by_uniqueness(preprocessed_df)
    sorted_df['question'] = sorted_df.apply(create_question, axis=1, args=(words_without_duplicates_but_multiple_meanings,))
    write_df(sorted_df, args.output_dir)

parser = argparse.ArgumentParser()
parser.set_defaults(function=encode)
parser.add_argument("--output_dir", dest='output_dir', default=None)
