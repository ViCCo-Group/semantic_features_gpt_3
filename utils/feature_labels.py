import pandas as pd 
import spacy 
import os 

DATA_DIR = os.getenv('DATA_DIR')

nlp = spacy.load('en_core_web_sm')

shapes = pd.read_csv(f'{DATA_DIR}/label_vocab/shapes.txt', names=['word'])['word'].to_list()   
sizes = pd.read_csv(f'{DATA_DIR}/label_vocab/sizes.txt', names=['word'])['word'].to_list()
texture = pd.read_csv(f'{DATA_DIR}/label_vocab/texture.txt', names=['word'])['word'].to_list()
color = pd.read_csv(f'{DATA_DIR}/label_vocab/colors.txt', names=['word'])['word'].to_list()

adjectives = shapes + color + texture + sizes

other = pd.read_csv(f'{DATA_DIR}/label_vocab/other_perceptual.txt', names=['word'])['word'].to_list()


def is_visual_perceptual(feature):
    if len(feature) >= 2:
        if feature[0].lemma_ == 'be' and feature[1].text in adjectives:
            return True

    if len(feature) >= 3:
        if feature[0].lemma_ == 'be' and feature[2].text in adjectives:
            return True

    if len(feature) >= 3:
        # could be smootie is made of fruit == built of (encyclopedic) or made of == consists of (visual)
        if feature[0].text == 'made' and feature[1].text in ['of', 'from']:
            return True

        if feature[1].text == 'made' and feature[2].text in ['of', 'from']:
            return True

    # TODO has a shape of
    # is bigger than a car
    # is part of stir - fries
    # smells good
    # has a thickness of 0
    # has four colors
    # is bouncy
    # is huge
    # is part of an orchestra -> not taxo
    # is used for fishing and is found near the beach
    # is shaped like a bowl
    # tastes fresh
    # is a part of a boat -> not taxo
    # is light in weight
    # has a smell of peppermint
    # has a flaky texture
    # looks like glass
    # has a gold or silver color
    # is sliced bread -> taxo
    # is straightened and bent
    # is for moving logs => functional is for VERB
    # is angular
    # is bronze
    # is mushy
    # is concave
    # is clear_transparent
    # is rectangular_square
    # is conical
    # is heart shaped
    # is waxy
    # is noisy_loud
    # does smell_is smelly
    # is camouflaged
    # is y-shaped
    # is vanilla flavoured
    # is spiral shaped
    # is muddy
    # has a strong flavour
    # does smell good_nice
    # has a distinctive smell
    # is cuboid
    # is scratchy

def is_taxonomic(feature):
    if len(feature) >= 2:
        if feature[0].text in ['a', 'an', 'the'] and feature[1].tag_.startswith('N'):
            return True 

        if feature[0].lemma_ == 'be' and feature[1].tag_.startswith('N'):
            return True

    if len(feature) >= 3:
        if feature[0].lemma_ == 'be' and feature[1].text in ['a', 'an', 'the'] and feature[2].tag_.startswith('N'):
            return True

    if len(feature) >= 4:
        if feature[0].lemma_ == 'be' and feature[1].text in ['a', 'an', 'the'] and feature[3].tag_.startswith('N'):
            return True   

def is_conceptual(feature):
    if feature[0].lemma_ == 'have':
        return True 

def is_functional(feature):
    if feature[0].lemma_ == 'be' and feature[1].text == 'used':
        return True 

    if feature[0].text == 'used':
        return True

def is_encyclopedic(feature):
    if len(feature) == 2:
        if feature[0].lemma_ == 'be' and feature[1].text not in adjectives and feature[1].text not in other and (feature[1].tag_.startswith('AD') or feature[1].tag_.startswith('JJ')):
            return True
    
    if feature[0].tag_.startswith('V'):
        return True

def is_other_perceptual(feature):
    if len(feature) >= 2:
        if feature[0].lemma_ == 'be' and feature[1].text in other:
            return True
        
    if len(feature) >= 3:
        if feature[0].lemma_ == 'be' and feature[2].text in other:
            return True

def get_feature_label(feature):
    feature = nlp(feature)
    
    if is_visual_perceptual(feature):
        return 'visual'
    if is_taxonomic(feature):
        return 'taxonomic'
    elif is_other_perceptual(feature):
        return 'other perceptual'
    elif is_conceptual(feature):
        return 'conceptual'
    elif is_functional(feature):
        return 'functional'
    elif is_encyclopedic(feature):
        return 'encyclopaedic'
    