import sys 
sys.path.append('..')
import os 
DATA_DIR = '../data'
os.environ['DATA_DIR'] = DATA_DIR

import pandas as pd 
from os.path import join as pjoin 

from semantic_norm_generator.decoding.decode import decode_answers

from utils.concepts import load_test_concepts, load_all_things_concepts, load_val_concepts


paths = [
    "../data/gpt_3_feature_norm/mcrae_priming/chatgpt-gpt3.5-turbo", # 1854 30
    "../data/gpt_3_feature_norm/mcrae_priming/gpt3-davinci", # 1854 30
    #"../data/gpt_3_feature_norm/mcrae_priming/gpt4", # 317 10
    "../data/gpt_3_feature_norm/cslb_priming/gpt3-davinci", # 1854 30
    "../data/gpt_3_feature_norm/mcrae_priming/claude" # 1854 30
]

for path in paths:
    df = pd.read_csv(pjoin(path, 'raw_answers.csv'), names=['concept', 'answer', 'concept_id', 'run_nr'], header=None)
    df['run_nr'] = df['run_nr'].astype(int)

    for concepts_name, concepts in [("test_concepts", load_test_concepts()), ("all_things_concepts", load_all_things_concepts())]:
        for runs in [10, 30]:
            print(f'Split and decode: {path} - {concepts_name} - {runs}')
            
            if runs == 30:
                run_nrs = list(run_nr for run_nr in range(1,31))
            else:
                run_nrs = list(df['run_nr'].unique())[:10] #[6, 23,  9,  7, 12, 27, 26, 17, 25, 22]

            df_filtered = df[(df['concept_id'].isin(concepts)) & (df['run_nr'].isin(run_nrs))]
            out_path = pjoin(path, concepts_name, str(runs))

            os.makedirs(out_path, exist_ok=True)
            raw_answers_path = pjoin(out_path, 'raw_answers.csv')
            df_filtered.to_csv(raw_answers_path, index=False)

            decoded_answers_path = pjoin(out_path, 'decoded_answers.csv')
            if os.path.exists(decoded_answers_path):
                print(f"skip decoding - {decoded_answers_path} exists")
                continue

            decoded_answers_df, rule_changes = decode_answers(df_filtered, False, True, True, out_path)
            decoded_answers_df.to_csv(decoded_answers_path, index=False)
            

