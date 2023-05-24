import utils.decoding.helper as helper
import spacy
nlp = spacy.load('en_core_web_sm')
from nltk.corpus import wordnet
import pandas as pd

def word_are_syn(word1, word2, pos):
    lemma2 = [syn for syn in wordnet.synsets(word1, pos=pos)][:1]
    lemma1 = [syn for syn in wordnet.synsets(word2, pos=pos)][:1]

    match = set(lemma1).intersection(set(lemma2))
    
    return len(match) != 0

def find_candidates(features):
    a_features = []
    used_features = []
    adj_features = []
    used_to_verb_feature = []
    used_to_noun_features = []
    used_to_verb_noun_feature = []
    made_features = []
    has_a_features = []

    # has a big mouth
    # has a large mouth
    has_a_adj_features = []
    has_a_adj_noun_features = []
    verb_noun_features = []
    verb_features = []

    other_verb_nouns = []

    for feature, count in features:
        tagged_feature = nlp(feature)
        
        if tagged_feature[0].lemma_ == 'be':
            # is a plant
            # are plants
            if len(tagged_feature) == 3 and tagged_feature[1].tag_ == 'DT':
                noun = tagged_feature[2]
                a_features.append((feature, noun.lemma_, count, noun.text))
            elif len(tagged_feature) == 2 and helper.word_is_noun(tagged_feature[1]):
                noun = tagged_feature[1]
                a_features.append((feature, noun.lemma_, count, noun.text))
        
            # is used for sport
            # is used in sport
            # is used in a sport
            elif len(tagged_feature) > 2 and tagged_feature[1].lemma_ == 'use' and \
            tagged_feature[2].text in ['as', 'at', 'in', 'for', 'on']:
                if tagged_feature[3].tag_ == 'DT' and len(tagged_feature) == 5:
                    noun = tagged_feature[4]
                    used_features.append((feature, noun.lemma_, count, noun.text))
                elif helper.word_is_noun(tagged_feature[3]) and len(tagged_feature) == 4:
                    noun = tagged_feature[3]
                    used_features.append((feature, noun.lemma_, count, noun.text))

            # is made from fruit
            # is made of fruits
            elif len(tagged_feature) > 2 and tagged_feature[1].lemma_ == 'make' and \
            tagged_feature[2].text in ['of', 'from']:
                if tagged_feature[3].tag_ == 'DT' and len(tagged_feature) == 5:
                    noun = tagged_feature[4]
                    made_features.append((feature, noun.lemma_, count, noun.text))
                elif helper.word_is_noun(tagged_feature[3]) and len(tagged_feature) == 4:
                    noun = tagged_feature[3]
                    made_features.append((feature, noun.lemma_, count, noun.text))

            # is used to store food
            # is used to keep food
            elif len(tagged_feature) > 4 and tagged_feature[1].lemma_ == 'use' and \
            tagged_feature[2].text == 'to':
                if helper.word_is_verb(tagged_feature[3]):
                    if len(tagged_feature) == 4:
                        # is used to save
                        # is used to store
                        verb = tagged_feature[3]
                        used_to_verb_feature.append((feature, verb.lemma_, count, verb.text))
                    elif len(tagged_feature) == 5 and helper.word_is_noun(tagged_feature[4]):
                        # is used to save money - is used to save cash
                        # is used to store money - is used save money
                        # is used to save money  - us used to store cash
                        noun = tagged_feature[4]
                        verb = tagged_feature[3]
                        used_to_noun_features.append((feature, noun.lemma_, count, noun.text))
                        used_to_verb_noun_feature.append((feature, verb.lemma_, count, verb.text))

            # is green
            # is greenish
            elif len(tagged_feature) == 2 and helper.word_is_adjective(tagged_feature[1]):
                noun = tagged_feature[1]
                adj_features.append((feature, noun.lemma_, count, noun.text))

        
        elif tagged_feature[0].lemma_ == 'have':
            # has a car
            if len(tagged_feature) == 3 and tagged_feature[1].tag_ == 'DT' and helper.word_is_noun(tagged_feature[2]):
                noun = tagged_feature[2]
                has_a_features.append((feature, noun.lemma_, count, noun.text))
            # has automobiles
            elif len(tagged_feature) == 2 and helper.word_is_noun(tagged_feature[1]):
                noun = tagged_feature[1]
                has_a_features.append((feature, noun.lemma_, count, noun.text))
            # has a large mouth 
            # has a big mouth
            # has big mouth
            elif len(tagged_feature) == 4 and tagged_feature[1].tag_ == 'DT' and helper.word_is_adjective(tagged_feature[2]) and\
                helper.word_is_noun(tagged_feature[3]):
                adjective = tagged_feature[2]
                noun = tagged_feature[3]
                has_a_adj_features.append((feature, adjective.lemma_, count, adjective.text))
                has_a_adj_noun_features.append((feature, noun.lemma_, count, noun.text))

        # protects eyes - protects an eye
        # protects the head - saves the head
        elif helper.word_is_verb(tagged_feature[0]) and tagged_feature[0].lemma_ != 'be':
            noun = None
            verb = None
            if len(tagged_feature) == 3 and tagged_feature[1].tag_ == 'DT' and helper.word_is_noun(tagged_feature[2]):
                noun = tagged_feature[2]
                verb = tagged_feature[0]
            elif len(tagged_feature) == 2 and helper.word_is_noun(tagged_feature[1]):
                noun = tagged_feature[1]
                verb = tagged_feature[0]
            if noun:
                verb_noun_features.append((feature, noun.lemma_, count, noun.text))
            if verb:
                verb_features.append((feature, verb.lemma_, count, verb.text))

            # is found in forests
            # grows in forests, grows in a forest, grows in the forest
            elif len(tagged_feature) > 1 and tagged_feature[1].text in ['as', 'at', 'in', 'for', 'on']:
                if len(tagged_feature) == 4 and tagged_feature[2].tag_ == 'DT':
                    noun = tagged_feature[3]
                    other_verb_nouns.append((feature, noun.lemma_, count, noun.text))
                elif len(tagged_feature) == 3 and helper.word_is_noun(tagged_feature[2]):
                    noun = tagged_feature[2]
                    other_verb_nouns.append((feature, noun.lemma_, count, noun.text))

    return [(a_features, wordnet.NOUN, False), (used_features, wordnet.NOUN, False),
            (adj_features, wordnet.ADJ, False), (used_to_verb_feature, wordnet.VERB, False), 
            (used_to_noun_features, wordnet.NOUN, True), (used_to_verb_noun_feature, wordnet.VERB, True), 
            (made_features, wordnet.NOUN, False), (has_a_features, wordnet.NOUN, False),
            (verb_noun_features, wordnet.NOUN, True), (verb_features, wordnet.VERB, True),
            (has_a_adj_features, wordnet.ADJ, True), (has_a_adj_noun_features, wordnet.NOUN, True),
            (other_verb_nouns, wordnet.NOUN, True)]

# is used in war, is used in wars, is used in warfare, is used for war, is used for wars
def get_synonyms_from_candidates(candidates):
    clusters = []
    for group, pos, only_word_allowed_to_differ in candidates:
        for i, feature_tuple1 in enumerate(group):
            for feature_tuple2 in group[i+1:]:
                feature1 = feature_tuple1[0]
                noun1_lemma = feature_tuple1[1]
                noun1_text = feature_tuple1[3]
                count1 = feature_tuple1[2]
                feature2 = feature_tuple2[0]
                noun2_lemma = feature_tuple2[1]
                count2 = feature_tuple2[2]
                noun2_text = feature_tuple2[3]

                if only_word_allowed_to_differ:
                    tokenized_feature1 = helper.tokenize(feature1)
                    tokenized_feature1.remove(noun1_text)
                    tokenized_feature2 = helper.tokenize(feature2)
                    tokenized_feature2.remove(noun2_text)

                    stopwords = ['a', 'an', 'the']
                    for stopword in stopwords:
                        if stopword in tokenized_feature1:
                            tokenized_feature1.remove(stopword)
                        if stopword in tokenized_feature2:
                            tokenized_feature2.remove(stopword)

                    if tokenized_feature1 != tokenized_feature2:
                        continue

                if word_are_syn(noun1_lemma, noun2_lemma, pos):
                    not_found = True
                    #feature_with_word1 = set(group[noun1].items())
                    features = set([(feature1,count1) , (feature2, count2)])
                        
                    if len(features) > 1:
                        for cluster in clusters:
                            for feature in features:
                                if feature in cluster:
                                    cluster = cluster.union(features)
                                    not_found = False
                                    break
                            
                        if not_found:
                            clusters.append(features)
    return clusters 

def join_df(df, syns, output_dir):
    rows = []
    print(syns)
    for group in syns:
        max_count = 0
        max_feature = None
        for feature, count in group:
            if count > max_count:
                max_count = count
                max_feature = feature
        group.remove((max_feature, max_count))
        for feature, _ in group:
            rows.append({'decoded_feature': feature, 'syn_feature': max_feature})
    syn_df = pd.DataFrame(rows)
    #syn_df.to_csv(f'{output_dir}/rules/syns.csv')
    df = df.merge(syn_df, how='left', on='decoded_feature')
    df.loc[df['syn_feature'].isnull(), 'syn_feature'] = df['decoded_feature']
    df = df.drop(['decoded_feature'], axis=1)
    df = df.rename({'syn_feature': 'decoded_feature'}, axis=1)
    return df
    

def find_synonyms(df, output_dir):
    features = df.groupby('decoded_feature', as_index=False).agg({'concept_id': 'count'})
    features = [(row[1]['decoded_feature'], row[1]['concept_id']) for row in features.iterrows()]
    cands = find_candidates(features)
    if cands:
        syns = get_synonyms_from_candidates(cands)
        df = join_df(df, syns, output_dir)                                  
    return df  