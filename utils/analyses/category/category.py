
from utils.vectorization import vectorize_concepts
from utils.data import load_things, load_gpt, load_cslb, load_sorting, load_cslb_count_vec, load_mcrae, generate_concepts_to_keep, match_behv_sim, load_behav

def get_categories():
    things_df = load_things()
    animal = ['moose', 'cow', 'alligator', 'bear', 'beaver', 'camel', 'cheetah', 'chipmunk', 'deer', 'dog', 'donkey', 'elephant', 'frog', 'giraffe', 'gorilla', 'goat', 'fox', 'hamster', 'hyena', 'horse', 'iguana', 'lamb', 'leopard', 'lion', 'otter', 'panther', 'pig', 'platypus', 'rabbit', 'raccoon', 'rat', 'sheep', 'skunk', 'squirrel', 'tiger', 'toad', 'zebra', 'rattlesnake', 'bat1', 'seal', 'cat', 'pony', 'calf1', 'mouse1', 'porcupine']
    clothing = ['jeans', 'helmet', 'cloak', 'apron', 'swimsuit', 'sweater', 'scarf', 'robe', 'jacket', 'dress', 'bra', 'blouse', 'belt', 'coat', 'shirt']
    bird = ['eagle', 'flamingo', 'hawk', 'ostrich', 'owl', 'pelican', 'penguin', 'pigeon', 'seagull', 'swan', 'chicken1', 'turkey', 'goose', 'duck']
    vehicle = ['boat', 'car', 'bus', 'helicopter', 'jeep', 'limousine', 'motorcycle', 'ship', 'taxi', 'tractor', 'train', 'yacht', 'trolley']
    fruit = list(things_df[things_df['category'] == 'fruit']['concept_id'].unique()) #['broccoli', 'carrot', 'corn', 'bean', 'tomato', 'celery', 'asparagus']
    musical = list(things_df[things_df['category'] == 'musical instrument']['concept_id'].unique()) #['hose', 'knife', 'broom', 'shovel', 'scissors', 'axe', 'crowbar']
    categories = [('animals', animal, 'Land animals'), ('clothing', clothing, 'Clothing'), ('bird', bird, 'Birds'), ('vehicle', vehicle, 'Vehicles'), ('fruit', fruit, 'Fruits'), ('musical', musical, 'Musical instruments')]
    
    all_short_categories = animal[:7] + clothing[:7] + bird[:7] + vehicle[:7] + fruit[:7] + musical[:7]

    return categories, all_short_categories, animal