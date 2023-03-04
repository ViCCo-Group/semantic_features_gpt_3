import os 
from os.path import join as pjoin 

import pandas as pd
import numpy as np
from IPython.display import display


DATA_DIR = os.environ['DATA_DIR']

def read_ground_truth(sample_id):
    truth = pd.read_csv(pjoin(DATA_DIR, 'quality', f'sample_with_true_source_{sample_id}.csv'))
    #truth = truth[truth['name'] != 'gpt_3_filtered_overlap']
    return truth

def read_prediction_0():
    pred = pd.read_excel(pjoin(DATA_DIR, 'quality', f'sample_0_pred.xlsx'), header=1)[['index', 'judgment']]
    pred['judgment'] = pd.to_numeric(pred['judgment'])
    return pred

def read_prediction_1():
    pred = pd.read_csv(pjoin(DATA_DIR, 'quality', f'sample_1_pred.csv'), header=0)[['index', 'judgment']]
    pred["judgment"] = pd.to_numeric(pred["judgment"])
    return pred

def merge_truth_and_pred(truth, pred):
    merge = truth.merge(pred, on='index')
    return merge

def load_all_quality_judgements():
    truth0 = read_ground_truth(0)
    judge0= read_prediction_0()
    merge0 = merge_truth_and_pred(truth0, judge0)

    truth1 = read_ground_truth(1)
    judge1 = read_prediction_1()
    merge1 = merge_truth_and_pred(truth1, judge1)

    merge = pd.concat([merge0, merge1])
    merge_unique = merge.drop_duplicates(['concept_id', 'feature'])

    cslb = merge_unique[merge_unique['name'] == 'cslb'].sample(480, replace=False)
    mcrae =  merge_unique[merge_unique['name'] == 'mcrae'].sample(480, replace=False)
    gpt = merge_unique[merge_unique['source'] == 'gpt'].sample(480, replace=False).drop(columns=['name']).rename(columns={'source': 'name'})

    result = pd.concat([cslb, mcrae, gpt])

    return result


def calc_relative_number_of_sensible_features(df):
    # calc relative number of features that are marked as sensible (1 or 2)
    # use bootstrap to calculate 95% CI

    def calc_for_sample(judgements, n_features):
        n_features_marked_as_sensible = len([judgement for judgement in judgements if judgement == 1 or judgement == 2])
        acc = n_features_marked_as_sensible / n_features * 100
        return acc
    
    judgements = df['judgment']
    n_features = df.shape[0]
    
    values = []
    for i in range(1000):
        judgements_bootstrap_sample = np.random.choice(judgements, n_features, replace=True)
        values.append(calc_for_sample(judgements_bootstrap_sample, n_features))

    sample_value = calc_for_sample(judgements, n_features)

    return (sample_value, np.asarray(values).std() * 1.96)

def show(result):
    rows = []
    for source in ['human', 'gpt']:
        result_source = result[result['source'] == source]
        result_source = result_source.drop_duplicates(['concept_id', 'feature', 'source'])
        sample_acc, ci = calc_relative_number_of_sensible_features(result_source)
        rows.append({'source': source, 'acc': sample_acc, 'ci': ci, 'n': result_source.shape[0]})
    
    df_human_gpt = pd.DataFrame(rows)
    df_human_gpt = df_human_gpt.round(2)
    display(df_human_gpt)
    
    rows = []
    for name in ['gpt_3_filtered', 'gpt_3_unfiltered', 'mcrae', 'cslb']:
        result_source = result[result['name'] == name]
        result_source = result_source.drop_duplicates(['concept_id', 'feature', 'name'])
        sample_acc, ci = calc_relative_number_of_sensible_features(result_source)
        rows.append({'source': name, 'acc': sample_acc, 'ci': ci, 'n': result_source.shape[0]})
    
    df_fine = pd.DataFrame(rows)
    df_fine = df_fine.round(2)
    display(df_fine)