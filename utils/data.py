import scipy.io
import pandas as pd
import math 
from os.path import join as pjoin
import os 

DATA_DIR = os.getenv('DATA_DIR')

def similarity_join(values):
    feature = ''
    try:
        for i in range(13):
           feature += values.iloc[i] + ';'
    except:
        pass
    return feature

def join_to_string(values):
    return ';'.join(values)

def load_things():
    return pd.read_csv(f'{DATA_DIR}/things/Wordlist_ratings-Final.csv')

def load_gpt(min_amount_runs_feature_occured, group_to_one_concept, min_amount_runs_feature_occured_within_concept, duplicates):
    # Read GPT3 features
    gpt_df = pd.read_csv(f'{DATA_DIR}/gpt_3_feature_norm/mcrae_priming/decoded_answers.csv')
    gpt_df = gpt_df.rename(columns={'decoded_feature': 'feature'})

    # keep features with 2 run occurence
    gpt_df = gpt_df[gpt_df['amount_runs_feature_occured'] >= min_amount_runs_feature_occured]
    gpt_df = gpt_df[gpt_df['amount_runs_feature_occured_within_concept'] >= min_amount_runs_feature_occured_within_concept]

    if not duplicates:
        gpt_df = gpt_df.drop_duplicates(['concept_id', 'feature'])

    if group_to_one_concept:
        gpt_df = gpt_df.groupby('concept_id', as_index=False).agg({'feature': join_to_string})
    return gpt_df

def load_mcrae(group_to_one_concept, duplicates):
    # Read McRae features
    print(DATA_DIR)
    mc_df = pd.read_excel(pjoin(DATA_DIR, 'mcrae/CONCS_FEATS_concstats_brm.xlsx'))
    mc_df = mc_df.rename(columns={'Concept': 'concept_id', 'Feature': 'feature'})
    def remove_beh(feature):
        if feature.startswith('beh') or feature.startswith('inbeh'):
            feature_split = feature.split('_')
            clean_feature_split = feature_split[2:]
            return '_'.join(clean_feature_split)
        return feature
    mc_df['feature'] = mc_df['feature'].apply(remove_beh)
    mc_df['feature'] = mc_df['feature'].apply(lambda feature: feature.replace('_', ' '))

    if duplicates:
        rows_with_duplicates = []
        for row in mc_df.iterrows():
            frequency = row[1].Prod_Freq
            for i in range(frequency):
                rows_with_duplicates.append(row[1])
        mc_df = pd.DataFrame(rows_with_duplicates)

    if group_to_one_concept:
        mc_df = mc_df.groupby('concept_id', as_index=False).agg({'feature': join_to_string})
    
    # Mapping Mc Rae Things concepts
    mapping_df = pd.read_csv(f'{DATA_DIR}/mapping.csv')
    for row in mapping_df.itertuples():
        mc_df.loc[mc_df['concept_id'] == row.mcrae_concept, 'concept_id'] = row.things_concept_id

    return mc_df

def load_behav():
    # Read similarity data from THINGS behaviourial data
    behv_sim = scipy.io.loadmat(pjoin(DATA_DIR, 'things/spose_similarity.mat'))['spose_sim']
    return behv_sim

def load_dimension_labels(n_dimensions):
    if n_dimensions == 49:
        labels = scipy.io.loadmat(pjoin(DATA_DIR, f'things/labels_{n_dimensions}.mat'))['labels']
        labels = [label[0][0] for label in labels]
    elif n_dimensions == 66:
        labels = open(pjoin(DATA_DIR, f'things/labels_{n_dimensions}.txt')).read().splitlines()
    return labels

def load_sorting():
    # Read things concept sorting
    sorting_df = pd.read_csv(pjoin(DATA_DIR, 'things/unique_id.txt'), header=None, names=['concept_id'])
    return sorting_df

def load_cslb(group_to_one_concept, remove_merged_features=False):
    cslb_df = pd.read_csv(pjoin(DATA_DIR, 'cslb/norms.dat'), sep='\t')
    cslb_df = cslb_df.rename(columns={'concept': 'concept_id', 'feature type': 'label'})
    cslb_df = cslb_df[['concept_id', 'feature', 'label']]
    if group_to_one_concept:
        cslb_df = cslb_df.groupby('concept_id', as_index=False).agg({'feature': join_to_string})

    mapping_df = pd.read_csv(f'{DATA_DIR}/mapping.csv')
    for row in mapping_df.itertuples():
        cslb_df.loc[cslb_df['concept_id'] == row.cslb_concept, 'concept_id'] = row.things_concept_id

    if remove_merged_features:
        cslb_df['feature'] = cslb_df['feature'].apply(lambda feature: feature.split('_')[0])
    return cslb_df 

def load_cslb_count_vec():
    df = pd.read_csv(f'{DATA_DIR}/cslb/feature_matrix.dat', header=0, sep='\t')
    df = df.rename(columns={'Vectors': 'concept_id'})
    mapping_df = pd.read_csv(f'{DATA_DIR}/mapping.csv')
    for row in mapping_df.itertuples():
        df.loc[df['concept_id'] == row.cslb_concept, 'concept_id'] = row.things_concept_id
    df = df.set_index('concept_id')
    return df

def generate_concepts_to_keep(gpt_df, mc_df=None, clsb_df=None, strategy='intersection'):
    concepts_to_keep = set(gpt_df['concept_id'])

    if strategy == 'exclusive_gpt_concepts':
        if mc_df is not None:
            concepts_to_keep = concepts_to_keep.difference(set(mc_df['concept_id']))
        if clsb_df is not None:
            concepts_to_keep = concepts_to_keep.difference(set(clsb_df['concept_id']))
    elif strategy == 'intersection':
        if mc_df is not None:
            concepts_to_keep = concepts_to_keep.intersection(set(mc_df['concept_id']))
        if clsb_df is not None:
            concepts_to_keep = concepts_to_keep.intersection(set(clsb_df['concept_id']))
    elif strategy == 'intersection_cslb_and_difference_mc':
        concepts_to_keep = concepts_to_keep.intersection(set(clsb_df['concept_id']))
        concepts_to_keep = concepts_to_keep.difference(set(mc_df['concept_id']))
    elif strategy == 'intersection_mc_and_difference_cslb':
        concepts_to_keep = concepts_to_keep.intersection(set(mc_df['concept_id']))
        concepts_to_keep = concepts_to_keep.difference(set(clsb_df['concept_id']))
    elif strategy == 'exclusive_cslb_concepts':
        concepts_to_keep = set(clsb_df['concept_id'])
        if mc_df is not None:
            concepts_to_keep = concepts_to_keep.difference(set(mc_df['concept_id']))
        if gpt_df is not None:
            concepts_to_keep = concepts_to_keep.difference(set(gpt_df['concept_id']))
    elif strategy == 'exclusive_mc_concepts':
        concepts_to_keep = set(mc_df['concept_id'])
        if clsb_df is not None:
            concepts_to_keep = concepts_to_keep.difference(set(clsb_df['concept_id']))
        if gpt_df is not None:
            concepts_to_keep = concepts_to_keep.difference(set(gpt_df['concept_id']))

    print('Amount of concepts to keep: %s' % str(len(concepts_to_keep)))
    return concepts_to_keep

def generate_concepts_to_keep2(feature_norms):
    concepts_to_keep = set(feature_norms[feature_norms.keys()[0]]['concept_id'])
    for name, feature_norm in feature_norms.items():
        concepts_to_keep = concepts_to_keep.intersection(set(feature_norm['concept_id']))
    return concepts_to_keep


def match_behv_sim(behv_sim, concepts_to_keep, sorting_df):
    concept_positions_to_keep = [sorting_df.index[sorting_df['concept_id'] == concept].tolist()[0] for concept in concepts_to_keep]
    concept_positions_to_keep = sorted(concept_positions_to_keep)
    behv_sim_matched = behv_sim[concept_positions_to_keep, :]
    behv_sim_matched = behv_sim_matched[:, concept_positions_to_keep]
    return behv_sim_matched

def match_mc(mc_df, concepts_to_keep):
    mc_df = mc_df[mc_df['concept_id'].isin(concepts_to_keep)]
    return mc_df

def match_gpt(gpt_df, concepts_to_keep):
    gpt_df = gpt_df[gpt_df['concept_id'].isin(concepts_to_keep)]
    return gpt_df

def match_cslb(cslv_df, concepts_to_keep):
    cslv_df = cslv_df[cslv_df['concept_id'].isin(concepts_to_keep)]
    return cslv_df

def export_matched_data(gpt_df, mc_df, behv_sim, clsb_df):
    data_dir = './evaluation/check_data'
    gpt_df.to_csv('%s/gpt_norms_only_matching.csv' % data_dir, index=False)
    pd.DataFrame(behv_sim).to_csv('%s/behavioral_similarity_only_matching.csv' % data_dir, index=False)
    if mc_df is not None:
        mc_df.to_csv('%s/mcrae_similarity_only_matching.csv' % data_dir, index=False)
    if clsb_df is not None:
        clsb_df.to_csv('%s/cslb_similarity_only_matching.csv' % data_dir, index=False)

def load_dimension_embeddings(n_dimensions):
    concept_ids = list(load_sorting()['concept_id'])
    if n_dimensions == 49:
        sep = ' '
    else:
        sep = '\t'
    
    dimensions = pd.read_csv(f'{DATA_DIR}/things/spose_embedding_{n_dimensions}d_sorted.txt', sep=sep, header=None, names=load_dimension_labels(n_dimensions), index_col=False)

    dimensions.index = concept_ids
    return dimensions

def load_bert(group_to_one_concept):
    def calc_pred(row):
        return math.exp(row.act_yes) / (math.exp(row.act_yes) + math.exp(row.act_no))
    
    def calc_pred2(row):
        return math.exp(row.act_no) / (math.exp(row.act_yes) + math.exp(row.act_no))
    

    df = pd.read_csv(f'{DATA_DIR}/bhatia/data.csv')
    df['prediction_yes'] = df.apply(calc_pred, axis=1)
    df['prediction_no'] = df.apply(calc_pred2, axis=1)
    df = df[df['prediction_yes'] > df['prediction_no']]

    df = df[['item', 'feature']]
    df = df.rename(columns={'item': 'concept_id'})
    mapping_df = pd.read_csv(f'{DATA_DIR}/mapping.csv')
    df['concept_id'] = df['concept_id'].apply(lambda concept_id: concept_id.replace(' ', '_'))
    for row in mapping_df.itertuples():
        df.loc[df['concept_id'] == row.cslb_concept, 'concept_id'] = row.things_concept_id

    if group_to_one_concept:
        df = df.groupby('concept_id', as_index=False).agg({'feature': join_to_string})

    return df

def load_data(mcrae=False, clsb=False, min_amount_runs_feature_occured=2, min_amount_runs_feature_occured_within_concept=2, group_to_one_concept=True, duplicates=False):
    gpt_df = load_gpt(min_amount_runs_feature_occured, group_to_one_concept, min_amount_runs_feature_occured_within_concept, duplicates)
    mc_df = None 
    clsb_df = None
    if mcrae:
        mc_df = load_mcrae(group_to_one_concept, duplicates)
    if clsb:
        clsb_df = load_cslb(group_to_one_concept)

    behv_sim = load_behav()

    return gpt_df, mc_df, behv_sim, clsb_df

def sort(df):
    sorted_df = load_sorting().reset_index().set_index('concept_id')
    df['concept_num'] = df.index.map(sorted_df['index'])
    df = df.sort_values(by='concept_num')
    df = df.drop('concept_num', axis=1)
    return df

