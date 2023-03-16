def generate_concepts_to_keep(feature_norms, strategy='intersection'):
    concepts_to_keep = set(feature_norms[list(feature_norms.keys())[0]]['concept_id'])
    for name, feature_norm in feature_norms.items():
        concepts_to_keep = concepts_to_keep.intersection(set(feature_norm['concept_id']))

    if strategy == 'excl_cslb_mcrae':
        all_things_concepts = set(feature_norms['GPT3-davinci-McRae']['concept_id'].unique())
        concepts_to_keep = all_things_concepts.difference(concepts_to_keep)

    print(f'{len(concepts_to_keep)} concepts are present in all feature norms')
    return concepts_to_keep


