import pandas as pd
import spacy 
import numpy as np
import concurrent.futures
from utils.decoding.rules.run import run_rules

from utils.decoding.synonyms import find_synonyms

nlp = spacy.load('en_core_web_sm')

def decode_answer_sentence(sentence):
    sentence = sentence.split('.')[0]
    features = sentence.split(',')
    return features

def split_answer_sentence(answer):
    if '.What' in answer or '. What' in answer or '.what' in answer or '. what' in answer:
        feature_sentence = answer.split('.')[0]
        features = feature_sentence.split(',')
    else:
        features = answer.split(',')
        # if . is missing, the feature could be stripped in the middle and make no sense
        features = features[:-1]
    return features  

def preprocess_feature(feature):
    feature = feature.strip()
    feature = feature.lower()
    return feature 

def create_rule_dfs_and_save(rule_changes, output_dir):
    rules = []
    for row in rule_changes:
        if row[2] not in rules:
            rules.append(row[2])
    for rule in rules:
        rule_rows = []
        for change in rule_changes:
            if rule == change[2]:
                rule_rows.append({'feature': change[0], 'changed_feature': change[1]})
        df = pd.DataFrame(rule_rows) 
        df.to_csv('%s/rules/%s.csv' % (output_dir, rule), index=False)


def split_and_decode_answer(answer, concept_id, run_nr, lemmatize):
    decoded_feature_list = []
    rules_list = []
    splitted_features = split_answer_sentence(answer)
    
    for raw_feature in splitted_features:
        preprocessed_feature = preprocess_feature(raw_feature)
        
        decoded_features, rule_changes = run_rules(preprocessed_feature, concept_id)
        rules_list += rule_changes

        if decoded_features:
            for decoded_feature in decoded_features:
                base_dict = {
                    'preprocessed_feature': preprocessed_feature,
                    'concept_id': concept_id,
                    'run_nr': run_nr
                }

                decoded_feature_dict = {**base_dict, **{
                    'decoded_feature': decoded_feature,
                }}

                decoded_feature_list.append(decoded_feature_dict)


    return decoded_feature_list, rules_list

def decode_batch(answers_df, batch_nr, lemmatize):
    decoded_rows = []
    rules_changes = []
    print('Start batch %s' % str(batch_nr))
    for row in answers_df.itertuples():
        print('Batch %s: #%s of %s' % (str(batch_nr), str(row.Index), answers_df.shape[0])) if row.Index % 100 == 0 else ''
        concept_id = row.concept_id
        answer = row.answer
        decoded_feature_list, batch_rules_changes = split_and_decode_answer(answer, concept_id, row.run_nr, lemmatize)
        decoded_rows += decoded_feature_list
        rules_changes += batch_rules_changes
    return decoded_rows, rules_changes

def filter_two_occ(run_nrs):
    amount_run_features_occured = len(set(run_nrs))
    return amount_run_features_occured

def decode_answers(answers_df, lemmatize, parallel, keep_duplicates_per_concept, output_dir):
    decoded_rows = []
    rules_changes = []

    if parallel:
        n = 8
        dfs = np.array_split(answers_df, n)
        with concurrent.futures.ProcessPoolExecutor(max_workers=n) as executor:
            future = [executor.submit(decode_batch, df, i, lemmatize) for i, df in enumerate(dfs)]
            for future in concurrent.futures.as_completed(future):
                result = future.result()
                decoded_rows += result[0]
                rules_changes += result[1]
    else:
        decoded_rows, rules_changes = decode_batch(answers_df, 1, lemmatize)
    df = pd.DataFrame(decoded_rows)

    # Drop duplicates of a feature within a concept and within a run -> same as when one human would write a feature twice 
    df = df.drop_duplicates(['concept_id', 'decoded_feature', 'run_nr'])

    # Sum amount of runs where the feature occured
    df['amount_runs_feature_occured'] = df['run_nr'].groupby(df['decoded_feature']).transform(filter_two_occ)

    # Sum amount of runs where the feature occured within a concept
    df['amount_runs_feature_occured_within_concept'] = df.groupby(['concept_id', 'decoded_feature'])['run_nr'].transform(filter_two_occ)

    # Calculate frequency of the feature in the total set of features
    #amount_all_features = df['decoded_feature'].shape[0]
    #df['feature_frequency'] = df['decoded_feature'].groupby(df['decoded_feature']).transform(lambda features: len(features) / amount_all_features * 100)

    df = find_synonyms(df, output_dir)

    return df, rules_changes