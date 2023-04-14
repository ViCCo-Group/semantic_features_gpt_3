from matplotlib import pyplot as plt 

def stats(df):
    n_features = df.shape[0]
    n_concepts = df['concept_id'].unique().shape[0]
    n_unique_features = df['feature'].unique().shape[0]
    mean_amount_features_per_concept = df.groupby('concept_id').agg({'feature': 'count'})['feature'].mean()
    mean_amount_of_concepts_per_feature = df.groupby('feature').agg({'concept_id': 'count'})['concept_id'].mean()
    print(f'Amount of features: {n_features}')
    print(f'Amount of concepts: {n_concepts}')
    print(f'Amount of unique features: {n_unique_features}')
    print(f'Mean amount of feature per concept: {mean_amount_features_per_concept}')
    if n_features != 0:
        print(f'Share of unique features to all features: {(n_unique_features/n_features)*100}')
    print(f'Mean amount of concepts per feature: {mean_amount_of_concepts_per_feature}')
    #TODO NoSF
    df_amount_concepts = df.groupby('feature', as_index=False).agg({'concept_id': 'count'}).rename(columns={'concept_id': 'concept_count'})
    df = df.merge(df_amount_concepts, on='feature', how='left')
    df = df[df['concept_count'] >= 3]
    mean_amount_shared_features_per_concept = df.groupby('concept_id').agg({'feature': 'count'})['feature'].mean()
    print(f'Mean amount of shared features per concept: {mean_amount_shared_features_per_concept}')
    
    return n_features, n_concepts, n_unique_features, mean_amount_features_per_concept, mean_amount_of_concepts_per_feature, mean_amount_shared_features_per_concept
