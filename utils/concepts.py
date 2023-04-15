import os 
from os.path import join as pjoin
DATA_DIR = os.getenv('DATA_DIR')

def load_test_concepts():
    with open(pjoin(DATA_DIR, 'concepts', 'test_concepts.txt'), 'r') as f:
        return f.read().splitlines()

def load_all_things_concepts():
    with open(pjoin(DATA_DIR, 'concepts', 'all_things_concepts.txt'), 'r') as f:
        return f.read().splitlines()

def load_val_concepts():
    with open(pjoin(DATA_DIR, 'concepts', 'val_concepts.txt'), 'r') as f:
        return f.read().splitlines()
