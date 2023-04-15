import sys 
sys.path.append('..')
import os 
DATA_DIR = '../data'
os.environ['DATA_DIR'] = DATA_DIR

import pandas as pd 
from utils.concepts import load_test_concepts, load_all_things_concepts, load_val_concepts
from os.path import join as pjoin 
from utils.decoding.decode import decode_answers


paths = [
    "../data/gpt_3_feature_norm/mcrae_priming/chatgpt-gpt3.5-turbo",
    "../data/gpt_3_feature_norm/mcrae_priming/gpt3-davinci",
    "../data/gpt_3_feature_norm/mcrae_priming/gpt4", # 317 10
    "../data/gpt_3_feature_norm/cslb_priming/gpt3-davinci" # 1854 10
]

for path in paths:
    print(path)
    df = pd.read_csv(pjoin(path, 'raw_answers.csv'), names=['concept', 'answer', 'concept_id', 'run_nr'])
    for concepts_name, concepts in [("test_concepts", load_test_concepts()), ("all_things_concepts", load_all_things_concepts()), ("validation_concepts", load_val_concepts())]:
        for runs in [10, 30]:
            if runs == 30:
                run_nrs = list(range(1,31))
            else:
                run_nrs = [6, 23,  9,  7, 12, 27, 26, 17, 25, 22]

            print(concepts_name)
            print(run_nrs)

            df_filtered = df[(df['concept_id'].isin(concepts)) & (df['run_nr'].isin(run_nrs))]
            out_path = pjoin(path, concepts_name, str(runs))

            os.makedirs(out_path, exist_ok=True)
            raw_answers_path = pjoin(out_path, 'raw_answers.csv')
            print(os.path.exists(raw_answers_path))
            if os.path.exists(raw_answers_path):
                print("skip")
                continue

            df_filtered.to_csv(raw_answers_path, index=False)
            print(df_filtered)
            try:
                decoded_answers_df, rule_changes = decode_answers(df_filtered, False, True, True, out_path)
                decoded_answers_df.to_csv(pjoin(out_path, 'decoded_answers.csv'), index=False)
            except Exception as e:
                continue

