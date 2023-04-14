import os 
from os.path import join as pjoin
DATA_DIR = os.getenv('DATA_DIR')

def load_test_concepts():
    pjoin(DATA_DIR, 'concepts', 'test')

def load_all_things_concepts():
    pjoin(DATA_DIR, 'concepts', 'all_things')

def load_val_concepts():
    pjoin(DATA_DIR, 'concepts', 'val')