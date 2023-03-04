from . import clean, extraction, transform
from copy import deepcopy

def transform_feature(feature, concept_id):
    rules = [
        # Removing apostrophe should be first -> it's -> is       
        ('replace_apos', transform.replace_apostrophe),
        # Removing pronom should be done before all other rules -> defined norm
        ('remove_pronom', transform.rule_remove_pronom),
        ('cut_that', transform.rule_remove_that),
        ('cut_when', transform.rule_remove_when),
        ('cut_but', transform.rule_remove_but),
        ('cut_such_as', transform.remove_such_as),
        ('cut_if', transform.rule_remove_if),
        ('cut_which', transform.rule_remove_which),
        ('remove_stopwords', transform.remove_stopwords),
        ('change_can', transform.rule_change_can),
        ('parenthesis', transform.remove_parenthesis),
        ('replace_number', transform.replace_digits),
        ('replace_british', transform.replace_british),
        ('remove_kind_of', transform.remove_kind_of),
        ('transform_used_for', transform.used),
        ('replace_plural_be_have', transform.replace_plural_be_have)
    ]
    if feature:
        rule_changes = []
    
        for rule in rules:
            feature_copy = deepcopy(feature)
            feature = rule[1](feature_copy)
            if feature_copy != feature: 
                rule_changes.append([feature_copy, deepcopy(feature), rule[0], concept_id])
        
        return feature, rule_changes
    return '', []

def extract_combinations(feature, concept_id):
    all_new_features = []
    all_rule_changes = []
    rules = [
        # And extraction should be first
        # it is a tree and a big plant -> it is a tree, it is big, it is a plant
        ('extract_ands', extraction.extract_and),
        ('extract_that', extraction.extract_that),
        ('extract_or_pronom_verb', extraction.extract_or_pronom_verb),
        ('extract_or_noun_adjective', extraction.extract_or_noun_adjective),
        ('extract_adjective_verb', extraction.extract_adjective_verb)
    ]

    def run_extractions(feature):
        old_feature = deepcopy(feature)
        new_features = []
        rule_changes = []
        for rule in rules:
            rule_name = rule[0]
            extracted_features = None
            extracted_features = rule[1](feature)
                
            if extracted_features:
                new_features += extracted_features
                for extracted_feature in extracted_features:
                    change = [old_feature, deepcopy(extracted_feature), rule_name, concept_id]
                    rule_changes.append(change)
        return new_features, rule_changes

    new_features1, rule_changes1 = run_extractions(feature) 
    all_rule_changes = all_rule_changes + rule_changes1

    for new_feature in new_features1:
        new_features2, rule_changes2 = run_extractions(new_feature) 
        if new_features2:
            all_rule_changes = all_rule_changes + rule_changes2
            all_new_features = all_new_features + new_features2
        else:
            all_new_features.append(new_feature)
    
    # it is a big tree -> extractions will be twice as the feature stays after extraction  
    all_new_features = list(set(all_new_features))
    return all_new_features, all_rule_changes

def clean_feature(feature, concept_id):
    concept = concept_id.replace('_', ' ')
    concept = ''.join([i for i in concept if not i.isdigit()])
    keep_feature = True
    rule_changes = []
    rules = [
        ('clean_properties_question', clean.rule_remove_properties_quesion),
        ('clean_underscores', clean.rule_remove_underscores),
        ('clean_one_word', clean.rule_remove_one_word_features),
        ('clean_remove_question_marks', clean.rule_remove_question_marks),
        ('rule_remove_not_pronom', clean.rule_remove_not_pronom),
        ('clean_remove_non_ascii', clean.rule_remove_non_ascii),
        ('clean_remove_same_noun', clean.remove_same_noun)
    ]
    old_feature = deepcopy(feature)

    for rule in rules:
        keep_feature = rule[1](feature, concept)
        if not keep_feature:
            change = [old_feature, '', rule[0], concept_id]
            rule_changes.append(change)
            break
    return keep_feature, rule_changes


def run_rules(preprocessed_feature, concept_id):
    decoded_features = []
    rule_changes = []

    keep_feature, clean_rule_changes = clean_feature(preprocessed_feature, concept_id)
    rule_changes += clean_rule_changes
    if keep_feature:
        extracted_features, extraction_rule_changes = extract_combinations(preprocessed_feature, concept_id)
        rule_changes += extraction_rule_changes
        if not extracted_features:
            extracted_features.append(preprocessed_feature)

        for feature in extracted_features:
            new_feature, feature_rule_changes = transform_feature(feature, concept_id) 
            rule_changes += feature_rule_changes
            not_single_word = clean.rule_remove_one_word_features(feature, concept_id)
            not_is_noun = clean.remove_same_noun(feature, concept_id)
            if not not_single_word:
                rule_changes.append([new_feature, '', 'remove_single_word', concept_id])
            if not not_is_noun:
                rule_changes.append([new_feature, '', 'remove_same_noun_last', concept_id])
            if new_feature and not_single_word and not_is_noun:
                decoded_features.append(new_feature)

    return decoded_features, rule_changes