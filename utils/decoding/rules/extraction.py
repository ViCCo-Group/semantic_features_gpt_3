from copy import deepcopy
import utils.decoding.helper as helper
import spacy
nlp = spacy.load('en_core_web_sm')

def extract_or_pronom_verb(feature):
    list_of_words = helper.tokenize(feature)
    tagged_feature = nlp(feature)
    extracted_list_of_words = []
    if 'or' in list_of_words:
        i = list_of_words.index('or')
        if len(list_of_words) > i+2:
            if tagged_feature[i+1].tag_ == 'PRP' and helper.word_is_verb(tagged_feature[i+2]):
                extracted_list_of_words.append(helper.combine_feature_list_to_string(list_of_words[i+1:]))
                extracted_list_of_words.append(helper.combine_feature_list_to_string(list_of_words[:i]))
    return extracted_list_of_words

def extract_or_noun_adjective(feature):
    list_of_words = helper.tokenize(feature)
    tagged_feature = nlp(feature)
    extracted_list_of_words = []
    if 'or' in list_of_words:
        i = list_of_words.index('or')
        first_feature = list_of_words[:i]
        if len(list_of_words) == i+2:
            previous_feature = tagged_feature[i+1] 
            if helper.word_is_noun(previous_feature) or helper.word_is_adjective(previous_feature):
                list_of_words[i-1] = tagged_feature[i + 1].text 
                list_of_words = list_of_words[:i]
                extracted_list_of_words.append(helper.combine_feature_list_to_string(list_of_words))
                extracted_list_of_words.append(helper.combine_feature_list_to_string(first_feature))
    return extracted_list_of_words

def extract_that(feature):
    list_of_words = helper.tokenize(feature)
    tagged_feature = nlp(feature)
    extracted_list_of_words = [] 
    if list_of_words and 'that' in list_of_words:
        i = list_of_words.index('that')
        first_feature = list_of_words[:i]
        if len(list_of_words) > i+1:
            previous_feature = tagged_feature[i-1]
            third_feature = tagged_feature[2]
            next_feature = tagged_feature[i+1]
            second_feature = tagged_feature[1]
            if helper.word_is_noun(previous_feature) and helper.word_is_verb(next_feature) and second_feature.lemma_ == 'be' and third_feature.tag_ == 'DT':
                pronom = list_of_words[0]
                extracted_feature = [pronom] + list_of_words[i+1:]
                extracted_list_of_words.append(helper.combine_feature_list_to_string(extracted_feature))
                extracted_list_of_words.append(helper.combine_feature_list_to_string(first_feature))
    return extracted_list_of_words

def extract_adjective_verb(feature):
    # it is a big tree -> a big tree -> a tree / is big
    list_of_words = helper.tokenize(feature)
    tagged_feature = nlp(feature)
    extracted_list_of_words = []
    if len(list_of_words) == 5:
        pronom = tagged_feature[0].text
        pronom_type = tagged_feature[0].tag_
        verb = tagged_feature[1].text
        article = tagged_feature[2].text
        article_typ = tagged_feature[2].tag_
        adjective = tagged_feature[3].text
        adjective_type = tagged_feature[3].tag_
        noun = tagged_feature[4].text

        if article_typ == 'DT' and pronom_type == 'PRP' and (verb == 'is' or verb == 'are'):           
            if adjective_type.startswith('AD') or adjective_type.startswith('JJ'):
                article = 'a'
                if helper.noun_with_vocal(noun):
                    article = 'an'
                feature1 = [pronom, verb, article, noun]
                extracted_list_of_words.append(helper.combine_feature_list_to_string(feature1))

                feature2 = [pronom, verb, adjective]
                extracted_list_of_words.append(helper.combine_feature_list_to_string(feature2))

                extracted_list_of_words.append(feature)

    return extracted_list_of_words

def extract_and(feature):
    # it is an animal and a dog -> is a dog 
    # they are animals and dogs -> are dogs
    # it is green and black -> is black
    # it eats grass and leaves and it is found in africa
    # it is long and thin
    extracted_features = []
    and_split = feature.split(' and ')

    if len(and_split) > 1:
        and_copy = deepcopy(and_split)

        def get_previous_case(feature):
            tagged_feature = nlp(feature)

            if len(tagged_feature) >= 3:
                # it is black
                # it is an animal
                # they are animals
                if tagged_feature[0].tag_ == 'PRP' and \
                    tagged_feature[1].lemma_ == 'be' and \
                        (helper.word_is_adjective(tagged_feature[2]) or tagged_feature[2].text == 'a' or tagged_feature[2].text == 'an' or helper.word_is_noun(tagged_feature[2])):
                        return [tagged_feature[0].text, tagged_feature[1].text]

                # it eats hay
                elif tagged_feature[0].tag_ == 'PRP' and\
                    helper.word_is_verb(tagged_feature[1]) and\
                        helper.word_is_noun(tagged_feature[2]):
                        return [tagged_feature[0].text, tagged_feature[1].text]

            if len(tagged_feature) >= 4:
                if tagged_feature[0].tag_ == 'PRP' and\
                    helper.word_is_verb(tagged_feature[2]) and\
                        (tagged_feature[3].text in ['by', 'of', 'for', 'in']):

                            # it is worn by men
                            # it is made of cream
                            # it is eaten by dogs
                            # it is used by humans
                            # it is used for surveillance
                            # it is used for dancing and 
                            # is is used in halls
                        if tagged_feature[3].text == 'for' and len(tagged_feature) > 5:
                            # it is used for flavouring food and drink
                            return [tagged_feature[0].text, tagged_feature[1].text, tagged_feature[2].text, tagged_feature[3].text, tagged_feature[4].text]
                        else:
                            return [tagged_feature[0].text, tagged_feature[1].text, tagged_feature[2].text, tagged_feature[3].text]

        last_complete = and_split[0]
        last_split = and_split[0]
        # add first split
        extracted_features.append(and_split[0])

        for i, split in enumerate(and_split):
            if i == 0:
                continue
            tagged_split = nlp(split)

            no_match = False
            
            # it is black and green -> it is green
            # they are animals and dogs -> they are dogs
            # it eats hay and grass -> it eats grass
            # it is worn by men and women -> it is worn by women
            # it is used for flavouring food and drink
            single_word = len(tagged_split) == 1 
            
            # it is a dog and a hound -> it is a hound
            article_word = len(tagged_split) == 2 and (tagged_split[0].text == 'a' or tagged_split[0].text == 'an')
            
            # it is made of ice cream and whipped cream and nuts and bananas and chocolate sauce
            # it is used for surveillance and military purposes
            # it is used for stirring and mixing and it is long and thin
            double_noun = len(tagged_split) == 2 and (helper.word_is_noun(tagged_split[0]) and helper.word_is_noun(tagged_split[1]))
    
            # it is black and it is green -> it is green
            # it is made of oats and its ingredients are wheat
            # it is a clear object and it can be used as a paperweight
            new_feature = (len(tagged_split) > 1 and tagged_split[0].tag_.startswith('PRP') and helper.word_is_verb(tagged_split[1])) or\
                          (len(tagged_split) > 2 and tagged_split[1].text == 'can' and helper.word_is_verb(tagged_split[2]))
            
            if new_feature:
                extracted_features.append(split)
                last_complete = split
                last_split = split
                and_copy.remove(split)
            elif single_word or article_word or double_noun:
                prev_words = get_previous_case(last_complete)
                if prev_words:
                    new_feature = prev_words + [split]
                    new_feature = helper.combine_feature_list_to_string(new_feature)
                    extracted_features.append(new_feature)
                    and_copy.remove(split)
                    last_split = new_feature 
                else:
                    # it is long and thin
                    # it lives in africa and asia and it can roar
                    no_match = True 
            else:
                # split has no match -> needs to be attached to last split
                # it is flat and it protects the driver and passengers from the wind
                # it has a motor and sails and is used for fishing and is found near the beach
                # it has oil and vinegar dressing and there are croutons
                no_match = True 

            if no_match:
                if last_split in extracted_features:
                    i = extracted_features.index(last_split)
                    extracted_features[i] = last_split + ' and ' + split
                else:
                    # it lives in africa and asia and it can roar
                    extracted_features.append(split)
                and_copy.remove(split)
                last_split = split 

    return extracted_features
