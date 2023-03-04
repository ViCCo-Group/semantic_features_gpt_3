import pandas as pd 
from data import generate_concepts_to_keep, load_gpt, load_cslb, load_mcrae
from labels import get_feature_label

n_intersection = 500
n_diff = 500

gpt_df = load_gpt(1,False,1,1,False)
clsb_df = load_cslb(False)
mc_df = load_mcrae(False, False)

intersection_concepts = generate_concepts_to_keep(gpt_df, mc_df, clsb_df, None, 'intersection')
#diff_concepts_gpt = generate_concepts_to_keep(gpt_df, mc_df, clsb_df, 'exclusive_gpt_concepts')
#diff_concepts_cslb = generate_concepts_to_keep(gpt_df, mc_df, clsb_df, 'exclusive_cslb_concepts')
diff_concepts_mc = generate_concepts_to_keep(gpt_df, mc_df, clsb_df, None, 'exclusive_mc_concepts')

#gpt_intersection_features = pd.DataFrame({'feature': gpt_df[gpt_df['concept_id'].isin(intersection_concepts)]['feature'].unique()}).sample(n_intersection)
#cslb_intersection_features = pd.DataFrame({'feature': clsb_df[clsb_df['concept_id'].isin(intersection_concepts)]['feature'].unique()}).sample(n_intersection)
mc_df_inter = pd.DataFrame({'feature': mc_df[mc_df['concept_id'].isin(intersection_concepts)]['feature'].unique()}).sample(n_intersection)

#gpt_diff_features = pd.DataFrame({'feature': gpt_df[gpt_df['concept_id'].isin(diff_concepts_gpt)]['feature'].unique()}).sample(n_diff)
#cslb_diff_features = pd.DataFrame({'feature': clsb_df[clsb_df['concept_id'].isin(diff_concepts_cslb)]['feature'].unique()}).sample(n_diff)
mc_df_exc = pd.DataFrame({'feature': mc_df[mc_df['concept_id'].isin(diff_concepts_mc)]['feature'].unique()}).sample(n_diff)

#gpt_df = pd.concat([gpt_intersection_features,gpt_diff_features])
#gpt_df['label'] = gpt_df['feature'].apply(get_feature_label)
#gpt_df.to_csv('gpt.csv', index=False)

#cslb_df = pd.concat([cslb_intersection_features,cslb_diff_features])
#cslb_df['label'] = cslb_df['feature'].apply(get_feature_label)
#cslb_df.to_csv('cslb.csv', index=False)

mc_df = pd.concat([mc_df_inter,mc_df_exc])
mc_df['label'] = mc_df['feature'].apply(get_feature_label)
mc_df.to_csv('mc.csv', index=False)