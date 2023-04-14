import pandas as pd
import os 

path = '/home/hannes/data/laptop_sync/arbeit/max_planck/projects/gpt3_semantic_features/semantic_features_gpt_3/data/gpt_3_feature_norm/mcrae_priming/chatgpt-gpt3.5-turbo/raw_answers.csv'
train_dir = '/home/hannes/data/laptop_sync/arbeit/max_planck/projects/gpt3_semantic_features/semantic_features_gpt_3/data/gpt_3_feature_norm/mcrae_priming/priming_examples'

df = pd.read_csv(path, names=['concept', 'answer', 'concept_id', 'run_nr'])

for train_file_name in os.listdir(train_dir):
    print(f'Check {train_file_name}')
    run_nr = int(train_file_name.split('_')[1].split('.')[0])
    train_df = pd.read_csv('%s/%s' % (train_dir, train_file_name))
    concepts_ids = list(train_df[:3]['concept'].unique())
    df = df.drop(df[(df['concept_id'].isin(concepts_ids)) & (df['run_nr'] == run_nr)].index)
    

# check run nr type
df.to_csv('raw_answer_cleaned.csv', index=False)