import pandas as pd 
from utils.concepts import load_test_concepts, load_all_things_concepts, load_val_concepts
from os.path import join as pjoin 
from utils.decoding.decode import decode_answers

paths = [
    "/home/hannes/data/laptop_sync/arbeit/max_planck/projects/gpt3_semantic_features/semantic_features_gpt_3/data/gpt_3_feature_norm/mcrae_priming/chatgpt-gpt3.5-turbo",
    "/home/hannes/data/laptop_sync/arbeit/max_planck/projects/gpt3_semantic_features/semantic_features_gpt_3/data/gpt_3_feature_norm/mcrae_priming/gpt3-davinci",
    "/home/hannes/data/laptop_sync/arbeit/max_planck/projects/gpt3_semantic_features/semantic_features_gpt_3/data/gpt_3_feature_norm/mcrae_priming/gpt4",
    "/home/hannes/data/laptop_sync/arbeit/max_planck/projects/gpt3_semantic_features/semantic_features_gpt_3/data/gpt_3_feature_norm/cslb_priming/chatgpt-gpt3.5-turbo",
    "/home/hannes/data/laptop_sync/arbeit/max_planck/projects/gpt3_semantic_features/semantic_features_gpt_3/data/gpt_3_feature_norm/cslb_priming/gpt3-davinci"
]

for path in paths:
    df = pd.read_csv(path, names=['concept', 'answer', 'concept_id', 'run_nr'])
    for concepts_name, concepts in [("test_concepts", load_test_concepts()), ("all_things_concepts", load_all_things_concepts()), ("validation_concepts", load_val_concepts())]:
        for runs in [10, 30]:
            if runs == 30:
                run_nrs = list(range())
            else:
                run_nrs = [] #TODO

            df_filtered = df[(df['concept_id'].isin(concepts)) & (df['run_nr'].isin(run_nrs))]
            out_path = f'{concepts}/{runs}'
            df_filtered.to_csv(pjoin(out_path, 'raw_answers.csv'), index=False)
            
            decoded_answers_df, rule_changes = decode_answers(df_filtered, False, True, True, out_path)
            decoded_answers_df.to_csv(pjoin(out_path, 'decoded_answers.csv'), index=False)

