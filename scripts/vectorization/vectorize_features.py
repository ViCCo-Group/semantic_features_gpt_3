import sys 
import os 
sys.path.append('../..')

DATA_DIR = '../../data'
os.environ['DATA_DIR'] = DATA_DIR

from utils.data import load_gpt, load_sorting
from utils.vectorization import vectorize_concepts

def vectorize(method):
    min_amount_runs_feature_occured = 5
    group_to_one_concept = True
    gpt_df = load_gpt(min_amount_runs_feature_occured, group_to_one_concept, 1, None, True)
    gpt_vec = vectorize_concepts(gpt_df, load_sorting(), 'bla', method)
    gpt_vec.T.to_csv(f"feature_concept_matrix_{method}_frequency.csv")

vectorize('tfidf')
vectorize('count')