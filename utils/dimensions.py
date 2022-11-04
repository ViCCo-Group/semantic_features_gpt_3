import pandas as pd
from copy import deepcopy


def normalize_object_dimension_weights(object_dimension_embeddings):
    # Normalize object dimension weights 
    dim_sums = object_dimension_embeddings.sum(axis=0)
    normalized_object_dimension_embeddings = object_dimension_embeddings.div(dim_sums)
    return normalized_object_dimension_embeddings

def compute_feature_weights_for_dims(object_feature_embeddings, object_dimension_embeddings, dims):
    # Compute feature weight for each dimension 
    weighted_features_for_all_dims = pd.DataFrame()
    for dim in dims:
        print(f'Run dimension: {dim}')
        df = deepcopy(object_feature_embeddings)
        dimension_values = object_dimension_embeddings.loc[:, dim]

        df_weighted_by_dim = df.mul(dimension_values, axis=0)
        df_summed_over_objects = df_weighted_by_dim.sum(axis=0).to_frame().sort_index()

        df_renamed = df_summed_over_objects.rename(columns={df_summed_over_objects.columns[0]: dim})
        weighted_features_for_all_dims = pd.concat([weighted_features_for_all_dims, df_renamed], axis=1)
    return weighted_features_for_all_dims

def normalize_feature_weights_across_dims(weighted_features_for_all_dims, dims):
    # Normalize feature weights across dimensions by substracting mean value based on all other dimensions
    normed_features = pd.DataFrame()
    for dim in dims:
        features_for_all_other_dims = weighted_features_for_all_dims.drop(dim, axis=1)
        mean_per_feature = features_for_all_other_dims.mean(axis=1)

        dim_values = weighted_features_for_all_dims.loc[:, dim]
        dim_values_normed = dim_values.subtract(mean_per_feature).to_frame()

        dim_values_normed = dim_values_normed.rename(columns={dim_values_normed.columns[0]: dim})
        normed_features = pd.concat([normed_features, dim_values_normed], axis=1)
    return normed_features

def calc_top_feature_per_dim(object_dimension_embeddings, object_feature_embeddings, dims):
    '''
    Calculate the top feature for each dimension by weighting the feature frequency with the object dimension weights.

    :param object_feature_embeddings: is a objectXfeature matrix with either frequency or a normalized count like tfidf
    :param object_feature_embeddings: is a objectXdimension matrix with dimension weights for each object
    :param dims: is a list of dimension names
    '''
    object_dimension_embeddings = normalize_object_dimension_weights(object_dimension_embeddings)
    weighted_features_for_all_dims = compute_feature_weights_for_dims(object_feature_embeddings, object_dimension_embeddings, dims)
    normed_features = normalize_feature_weights_across_dims(weighted_features_for_all_dims, dims)
    return normed_features

def matrix_to_top_list(df):
    df_list = pd.DataFrame()
    for dim in df.columns:
        dim_values = df.loc[:, [dim]].reset_index()
        dim_values = dim_values.rename(columns={dim_values.columns[0]: 'feature', dim_values.columns[1]: 'weight'})
        top = dim_values.sort_values(by='weight', ascending=False)[:20]
        top['dimension'] = dim
        df_list = pd.concat([df_list, top])
    return df_list