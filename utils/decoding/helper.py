from spacy.lang.en import English
import inflect as word

nlp = English()
tokenizer = nlp.tokenizer

def get_list_of_words(feature_spacy, lemma=False):
    if not lemma:
       return [token.text for token in feature_spacy]
    return [token.lemma_ for token in feature_spacy]

def combine_feature_list_to_string(feature, connection_string=' '):
    return connection_string.join(feature)

def noun_with_vocal(noun):
    noun = noun.lower()
    return noun.startswith('a') or noun.startswith('e') or noun.startswith('i') or noun.startswith('o') or noun.startswith('u')

def tokenize(feature):
    return [str(word) for word in tokenizer(feature)]
    #return feature.split(' ')

def noun_is_singular(noun):
    inflect = word.engine()
    return inflect.singular_noun(noun) is False

def word_is_noun(tagged_word):
    if tagged_word.tag_.startswith('NN'):
        return True

def word_is_verb(tagged_word):
    if tagged_word.tag_.startswith('V'):
        return True

def word_is_adjective(tagged_word):
    if tagged_word.tag_.startswith('AD') or tagged_word.tag_ == 'JJ':
        return True 