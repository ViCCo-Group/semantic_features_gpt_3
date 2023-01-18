import argparse
import pandas as pd
from utils.decoding.rules.run import decode_answers, create_rule_dfs_and_save

def run_decode(args):
    answers = pd.read_csv(args.answers)
    output_dir = args.output
    lemmatize = args.lemmatize
    parallel = args.parallel
    keep_duplicates_per_concept = args.keep_duplicates_per_concept
    decoded_answers_df, rule_changes = decode_answers(answers, lemmatize, parallel, keep_duplicates_per_concept, output_dir)
    decoded_answers_df.to_csv('%s/decoded_answers.csv' % output_dir, index=False)
    create_rule_dfs_and_save(rule_changes, output_dir)

parser = argparse.ArgumentParser()
parser.set_defaults(function=run_decode)
parser.add_argument("--answers", dest='answers')
parser.add_argument("--output", dest='output')
parser.add_argument("--parallel", dest='parallel', action='store_true')
parser.add_argument("--lemmatize", dest='lemmatize', action='store_true')
parser.add_argument("--keep_duplicates_per_concept", dest='keep_duplicates_per_concept', action='store_true')

args = parser.parse_args()
args.function(args)