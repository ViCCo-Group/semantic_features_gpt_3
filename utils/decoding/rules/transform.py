import utils.decoding.helper as helper
import spacy
import mlconjug3
import json 

nlp = spacy.load('en_core_web_sm')
default_conjugator = mlconjug3.Conjugator(language='en')


def rule_remove_pronom(feature):
    if feature:
        tagged_feature = nlp(feature)
        list_of_words = helper.tokenize(feature)

        if tagged_feature[0].tag_ == 'PRP':
            del list_of_words[0]
        
        #if len(list_of_words) >= 3:
        #    if tagged_feature[1].lemma_ == 'be' and tagged_feature[2].tag_.startswith('V'):
        #        del list_of_words[0]
        #    elif tagged_feature[1].lemma_ == 'be' and tagged_feature[2].tag_ == 'DT' and tagged_feature[3].tag_.startswith('N'):
        #        del list_of_words[0]

        feature = helper.combine_feature_list_to_string(list_of_words)

    return feature


def rule_remove_but(feature):
    list_of_words = helper.tokenize(feature)
    if list_of_words and 'but' in list_of_words:
        i = list_of_words.index('but')
        list_of_words = list_of_words[:i]
        feature = helper.combine_feature_list_to_string(list_of_words)
    return feature

def rule_remove_when(feature):
    list_of_words = helper.tokenize(feature)
    if list_of_words and 'when' in list_of_words:
        i = list_of_words.index('when')
        list_of_words = list_of_words[:i]
        feature = helper.combine_feature_list_to_string(list_of_words)
    return feature

def rule_remove_if(feature):
    list_of_words = helper.tokenize(feature)
    if list_of_words and 'if' in list_of_words:
        i = list_of_words.index('if')
        list_of_words = list_of_words[:i]
        feature = helper.combine_feature_list_to_string(list_of_words)
    return feature

def rule_remove_that(feature):
    list_of_words = helper.tokenize(feature)
    tagged_feature = nlp(feature)
    if list_of_words and 'that' in list_of_words:
        i = list_of_words.index('that')
        previous_feature = tagged_feature[i-1]
        if previous_feature.text == 'so':
            list_of_words = list_of_words[:i-1]
        elif previous_feature.tag_.startswith('NN'):
            list_of_words = list_of_words[:i]
        feature = helper.combine_feature_list_to_string(list_of_words)

    return feature

def rule_remove_which(feature):
    list_of_words = helper.tokenize(feature)
    tagged_feature = nlp(feature)
    if list_of_words and 'which' in list_of_words:
        i = list_of_words.index('which')
        previous_feature = tagged_feature[i-1]
        if previous_feature.tag_.startswith('NN'):
            list_of_words = list_of_words[:i]
            feature = helper.combine_feature_list_to_string(list_of_words)
    return feature

def remove_such_as(feature):
    # it is big such as -> it is big
    list_of_words = helper.tokenize(feature)
    if list_of_words and 'such' in list_of_words:
        i = list_of_words.index('such')
        if len(list_of_words) > i+1:
            next_feature = list_of_words[i+1]
            if next_feature == 'as':
                list_of_words = list_of_words[:i]

            feature = helper.combine_feature_list_to_string(list_of_words)
    return feature

def remove_parenthesis(feature):
    # it is (some) big -> it is big
    list_of_words = helper.tokenize(feature)
    if list_of_words and '(' in list_of_words:
        index_first_parenthesis = list_of_words.index('(')
        if ')' in list_of_words:
            index_second_parenthesis = list_of_words.index(')')
            list_of_words = list_of_words[:index_first_parenthesis] + list_of_words[index_second_parenthesis+1:]
        else:
            list_of_words = list_of_words[:index_first_parenthesis]

        feature = helper.combine_feature_list_to_string(list_of_words)
    return feature

def rule_change_can(feature):
    # it can be red -> it is red 
    list_of_words = helper.tokenize(feature)
    tagged_feature = nlp(feature)
    if list_of_words and 'can' in list_of_words:
        i = list_of_words.index('can')
        if len(list_of_words) > i+1:
            next_feature = tagged_feature[i+1]
            if next_feature.tag_.startswith('V'):
                try:
                    list_of_words[i+1] = default_conjugator.conjugate(next_feature.text).conjug_info['indicative']['indicative present']['3s']
                except:
                    pass
                del list_of_words[i]
                

            feature = helper.combine_feature_list_to_string(list_of_words)
    return feature

def remove_stopwords(feature):
    # is is usually big -> it is big
    list_of_words = helper.tokenize(feature)
    stopwords = ['also', 'usually', 'very', 'really']
    for stopword in stopwords:
        if list_of_words and stopword in list_of_words:
            i = list_of_words.index(stopword)
            del list_of_words[i]
    feature = helper.combine_feature_list_to_string(list_of_words)
    return feature

def used(feature):
    # used for hunting -> used to hunt
    # can be used to
    # is used to move -> used to move
    tagged_feature = nlp(feature)
    list_of_words = helper.tokenize(feature)
    verbs = ['used', 'worn']

    match = False 
    matched_start_pos = None
    for start_pos in [0, 1, 2]:
        if len(list_of_words) >= start_pos+3:
            if tagged_feature[start_pos].text in verbs and tagged_feature[start_pos + 1].text == 'for' and tagged_feature[start_pos + 2].tag_.startswith('V'):
                match = True
                matched_start_pos = start_pos
                break

    if match:
        list_of_words[matched_start_pos + 1] = 'to'
        list_of_words[matched_start_pos + 2] = tagged_feature[matched_start_pos + 2].lemma_
        feature = helper.combine_feature_list_to_string(list_of_words)
    return feature

def replace_digits(feature):
    # it has 1 wheel -> it has one wheel
    list_of_words = helper.tokenize(feature)
    numbers = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
    for number, number_string in enumerate(numbers):
        number = number + 1
        if str(number) in list_of_words:
            list_of_words[list_of_words.index(str(number))] = number_string
    
    feature = helper.combine_feature_list_to_string(list_of_words)
    return feature

def replace_apostrophe(feature):
    # it's big -> it is big
    list_of_words = helper.tokenize(feature)
    tagged_feature = nlp(feature)
    for value in [("'s", "is"), ("'re", "are")]:
        old_value = value[0]
        new_value = value[1]
        if old_value in list_of_words:
            pos = list_of_words.index(old_value)
            if tagged_feature[pos-1].tag_ == 'PRP' and tagged_feature[pos-1].text != 'one':
                list_of_words[pos] = new_value
    
    feature = helper.combine_feature_list_to_string(list_of_words)
    return feature

def replace_plural_be_have(feature):
    # 'moves' -> 'moves'
    # 'move' -> 'moves'
    # 'are green' -> 'is green'
    # 'are used to make' -> 'is used to make'

    list_of_words = helper.tokenize(feature)
    tagged_feature = nlp(feature)

    if tagged_feature[0].tag_.startswith('V'):
        list_of_words[0] = default_conjugator.conjugate(tagged_feature[0].lemma_).conjug_info['indicative']['indicative present']['3s']
    
    feature = helper.combine_feature_list_to_string(list_of_words)
    return feature

def replace_british(feature):
    list_of_words = helper.tokenize(feature)

    with open('../data/british_to_american.json') as vocab_file:
        vocab = json.load(vocab_file)
    
    for british_word in vocab:
        american_word = vocab[british_word]
        if british_word in list_of_words:
            pos = list_of_words.index(british_word)
            list_of_words[pos] = american_word
    
    feature = helper.combine_feature_list_to_string(list_of_words)
    return feature

def remove_kind_of(feature):
    # is a kind of clothing -> is a clothing
    # are a kind of pants -> are pants
    list_of_words = helper.tokenize(feature)
    tagged_feature = nlp(feature)
    if len(list_of_words) >= 5:
        if tagged_feature[0].lemma_ == 'be' and \
            tagged_feature[2].text in ['kind', 'type'] and \
                tagged_feature[3].text == 'of':
                noun = list_of_words[4]
                verb = list_of_words[0]

                new_feature = [verb]
                if helper.noun_is_singular(noun):
                    article = 'a'
                    if helper.noun_with_vocal(noun):
                        article = 'an'
                    new_feature.append(article)
                list_of_words = new_feature + list_of_words[4:]
        feature = helper.combine_feature_list_to_string(list_of_words)
    return feature
