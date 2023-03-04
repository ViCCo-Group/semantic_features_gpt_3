from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd 
from utils.data import load_sorting, generate_concepts_to_keep, match_behv_sim, load_cslb_count_vec, sort, load_data, generate_concepts_to_keep2

def vectorize_concepts(df, sorting_df, vectorizer):
    # Calculate Vectors of all concepts based on the features
    if vectorizer == 'binary':
        vectorizer = CountVectorizer(tokenizer=lambda feature_string: feature_string.split(';'), binary=True)
    elif vectorizer == 'tfidf':
        vectorizer = TfidfVectorizer(tokenizer=lambda feature_string: feature_string.split(';'))
    elif vectorizer == 'count':
        vectorizer = CountVectorizer(tokenizer=lambda feature_string: feature_string.split(';'), binary=False)

    sparse_matrix = vectorizer.fit_transform(df['feature'])
    doc_term_matrix = sparse_matrix.todense()
    pred_count_df = pd.DataFrame(doc_term_matrix, 
                    columns=vectorizer.get_feature_names(), 
                    index=df['concept_id'])

    if sorting_df is not None:
        sorted_df = sorting_df.reset_index().set_index('concept_id')
        pred_count_df['concept_num'] = pred_count_df.index.map(sorted_df['index'])
        pred_count_df = pred_count_df.sort_values(by='concept_num')
        pred_count_df = pred_count_df.drop('concept_num', axis=1)

    #for column in pred_count_df.columns:
    #    feature_frequency = df.loc[df['feature'] == column][0]
    #    pred_count_df[column] = pred_count_df[column] * feature_frequency

    #data_dir = './evaluation/check_data'
    #pred_count_df.to_csv('%s/vectorized_concepts_%s.csv' % (data_dir, name))
    return pred_count_df

def get_all_vectorized(vec = 'binary', intersection='intersection'):
    gpt_df, mc_df, behv_sim, cslb_df = load_data(True, True, 4, 1, True, True)

    gpt_vec = vectorize_concepts(gpt_df, load_sorting(), vec)
    cslb_vec = vectorize_concepts(cslb_df, load_sorting(), vec)
    mc_vec = vectorize_concepts(mc_df, load_sorting(), vec)

    if vec == 'count':
        cslb_vec = load_cslb_count_vec()

    if intersection:
        intersection_concepts = generate_concepts_to_keep(gpt_df, mc_df, cslb_df, 'intersection')
        gpt_vec = gpt_vec.loc[intersection_concepts]
        cslb_vec = cslb_vec.loc[intersection_concepts]
        mc_vec = mc_vec.loc[intersection_concepts]
        behv_sim = match_behv_sim(behv_sim, intersection_concepts, load_sorting())

    gpt_vec = sort(gpt_vec)
    cslb_vec = sort(cslb_vec)
    mc_vec = sort(mc_vec)
    
    return gpt_vec, cslb_vec, mc_vec, behv_sim


def get_all_vectorized2(feature_norms, behv_sim, vec = 'binary', intersection='intersection'):
    feature_norms_vec = {}
    sorting = load_sorting()
    intersection_concepts = None 

    if intersection:
        intersection_concepts = generate_concepts_to_keep2(feature_norms, 'intersection')
        behv_sim = match_behv_sim(behv_sim, intersection_concepts, load_sorting())

    for name, feature_norm in feature_norms:
        feature_norm_vec = vectorize_concepts(feature_norm, sorting, vec)

        if vec == 'count' and name == 'cslb':
            feature_norm_vec = load_cslb_count_vec()

        if intersection:
            feature_norm_vec = feature_norm_vec.loc[intersection_concepts]

        feature_norms_vec[name] = sort(feature_norm_vec)
        
    return feature_norms_vec, behv_sim

