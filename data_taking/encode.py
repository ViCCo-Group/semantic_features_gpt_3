import pandas as pd
import inflect as word
import helper as helper
import inflect as word

def encode_feature(feature):
    splitted_feature = feature.split('_')
    first_word = splitted_feature[0]
    transformed_word = None 

    if first_word in ['is', 'has', 'requires', 'lives', 'tastes', 'likes', 'grows', 'comes', 'smells']:
        transformed_word = 'it %s' % first_word 
    elif first_word in ['are', 'have']:
        transformed_word = 'they %s' % first_word
    elif first_word in ['a', 'an', 'used', 'found', 'made', 'eaten', 'worn', 'herded', 'hunted', 'like', 'associated']:
        transformed_word = 'it is %s' % first_word
    elif first_word in ['different']:
        transformed_word = 'it has %s' % first_word
    elif first_word in ['symbol']:
        transformed_word = 'it is a %s' % first_word
    else:
        transformed_word = 'it %s' % first_word

    splitted_feature[0] = transformed_word
    return ' '.join(splitted_feature)

def noun_with_vocal(noun):
    noun = noun.lower()
    return noun.startswith('a') or noun.startswith('e') or noun.startswith('i') or noun.startswith('o') or noun.startswith('u')

def noun_is_singular(noun):
    inflect = word.engine()
    return inflect.singular_noun(noun) is False

def create_base_question(concept, check_plural=False, is_singular=False, category=None):
    inflect = word.engine()
    sentence = 'What are the properties of '
    
    if (check_plural and noun_is_singular(concept)) or (is_singular and not check_plural):
        if noun_with_vocal(concept):
            sentence += 'an '
        else:
            sentence += 'a '

    sentence += concept
    
    if category:
        sentence += ' (%s)' % category

    sentence += '?' 
    return sentence

def create_question(row, check_plural=True, is_singular=False):
    concept = row.concept
    question = create_base_question(concept, check_plural, is_singular)
    return question

def add_sentence_column(df):
    df['question'] = df.apply(create_question, axis=1)
    df['answer'] = df.apply(create_answer, axis=1)
    return df

def create_answer(row):
    answer = ''
    features = row.feature.split(',')
    encoded_features = [encode_feature(feature) for feature in features]
    encoded_features = [feature for feature in encoded_features if feature]
    n_features = len(encoded_features)
    for i, feature in enumerate(encoded_features):
        if i == 0:
            feature = feature.capitalize()
        answer += feature
        if i == n_features-2:
            answer += ' and '
        elif i == n_features-1:
            answer += '.'
        else:
            answer += ', '
    return answer


