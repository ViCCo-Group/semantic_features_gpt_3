from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd 

def vectorize_concepts(df, vectorizer):
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

    #for column in pred_count_df.columns:
    #    feature_frequency = df.loc[df['feature'] == column][0]
    #    pred_count_df[column] = pred_count_df[column] * feature_frequency

    #data_dir = './evaluation/check_data'
    #pred_count_df.to_csv('%s/vectorized_concepts_%s.csv' % (data_dir, name))
    return pred_count_df

