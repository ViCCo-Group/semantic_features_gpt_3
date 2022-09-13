import inflect as word
import random 

def encode_feature(concept, feature, check_plural, feature_seperator):
    splitted_feature = feature.split(feature_seperator)
    first_word = splitted_feature[0]
    transformed_word = None 

    pronom = 'it' 
    if not noun_is_singular(concept) and check_plural:
        pronom = 'they'

    if first_word in ['is', 'has', 'requires', 'lives', 'tastes', 'likes', 'grows', 'comes', 'smells']:
        transformed_word = f'{pronom} {first_word}' 
    elif first_word in ['are', 'have']:
        transformed_word = f'{pronom} {first_word}'
    elif first_word in ['a', 'an', 'used', 'found', 'made', 'eaten', 'worn', 'herded', 'hunted', 'like', 'associated']: 
        if pronom == 'it':
            verb = 'is'
        else:
            verb = 'are'
        transformed_word = f'{pronom} {verb} {first_word}'
    elif first_word in ['different']:
        if pronom == 'it':
            verb = 'has'
        else:
            verb = 'have'
        transformed_word = f'{pronom} {verb} {first_word}'
    elif first_word in ['symbol']:
        if pronom == 'it':
            verb = 'is'
        else:
            verb = 'are'
        transformed_word = f'{pronom} {verb} a {first_word}'
    else:
        transformed_word = f'{pronom} {first_word}'

    splitted_feature[0] = transformed_word
    return ' '.join(splitted_feature)

def noun_with_vocal(noun):
    noun = noun.lower()
    return noun.startswith('a') or noun.startswith('e') or noun.startswith('i') or noun.startswith('o') or noun.startswith('u')

def noun_is_singular(noun):
    inflect = word.engine()
    return inflect.singular_noun(noun) is False

def create_base_question(concept, check_plural=False, is_singular=False, category=None):
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

def add_sentence_column(df, max_n_features_per_concept=20, check_plural=True, feature_seperator=' '):
    df['question'] = df.apply(create_question, axis=1)
    df['answer'] = df.apply(create_answer, axis=1, max_n_features_per_concept=max_n_features_per_concept, check_plural=check_plural, feature_seperator=feature_seperator)
    return df

def create_answer(row, max_n_features_per_concept, check_plural, feature_seperator):
    answer = ''
    features = row.feature.split(',')
    if len(features) > max_n_features_per_concept:
        features = random.sample(features, max_n_features_per_concept)
    encoded_features = [encode_feature(row.concept, feature, check_plural, feature_seperator) for feature in features]
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


