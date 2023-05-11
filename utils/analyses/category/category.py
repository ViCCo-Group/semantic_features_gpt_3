from utils.data import load_things
import pandas as pd

def get_categories(intersection_concepts):
    things_df = load_things()
    things_df = things_df.rename(columns={'uniqueID': 'concept_id'})
    things_df = things_df[['concept_id', 'All Bottom-up Categories']].set_index('concept_id')
    things_df = things_df.loc[intersection_concepts].reset_index()

    new_rows = []
    for row in things_df.iterrows():
        cats = row[1]['All Bottom-up Categories'].split(',')
        for cat in cats:
            new_rows.append({'concept_id': row[1].concept_id, 'category': cat})
            
    things_df = pd.DataFrame(new_rows)

    animal = ['moose', 'cow', 'alligator', 'bear', 'beaver', 'camel', 'cheetah', 'chipmunk', 'deer', 'dog', 'donkey', 'elephant', 'frog', 'giraffe', 'gorilla', 'goat', 'fox', 'hamster', 'hyena', 'horse', 'iguana', 'lamb', 'leopard', 'lion', 'otter', 'panther', 'pig', 'platypus', 'rabbit', 'raccoon', 'rat', 'sheep', 'skunk', 'squirrel', 'tiger', 'toad', 'zebra', 'rattlesnake', 'bat1', 'seal', 'cat', 'pony', 'calf1', 'mouse1', 'porcupine']
    clothing = ['jeans', 'helmet', 'cloak', 'apron', 'swimsuit', 'sweater', 'scarf', 'robe', 'jacket', 'dress', 'bra', 'blouse', 'belt', 'coat', 'shirt']
    bird = ['eagle', 'flamingo', 'hawk', 'ostrich', 'owl', 'pelican', 'penguin', 'pigeon', 'seagull', 'swan', 'chicken1', 'turkey', 'goose', 'duck']
    vehicle = ['boat', 'car', 'bus', 'helicopter', 'jeep', 'limousine', 'motorcycle', 'ship', 'taxi', 'tractor', 'train', 'yacht', 'trolley']
    fruit = list(things_df[things_df['category'] == 'fruit']['concept_id'].unique()) #['broccoli', 'carrot', 'corn', 'bean', 'tomato', 'celery', 'asparagus']
    musical = list(things_df[things_df['category'] == 'musical instrument']['concept_id'].unique()) #['hose', 'knife', 'broom', 'shovel', 'scissors', 'axe', 'crowbar']
    categories = [('animals', animal, 'Land animals'), ('clothing', clothing, 'Clothing'), ('bird', bird, 'Birds'), ('vehicle', vehicle, 'Vehicles'), ('fruit', fruit, 'Fruits'), ('musical', musical, 'Musical instruments')]
    
    all_short_categories = animal[:7] + clothing[:7] + bird[:7] + vehicle[:7] + fruit[:7] + musical[:7]

    return categories, all_short_categories, animal