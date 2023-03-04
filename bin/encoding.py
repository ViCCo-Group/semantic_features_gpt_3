import argparse
from utils.encoding.encode_things import encode as encode_things
from utils.encoding.encode_mcrae import encode as encode_mcrae
from utils.encoding.encode_cslb import encode as encode_cslb

parser = argparse.ArgumentParser()
parser.set_defaults(function=encode_things)
parser.add_argument("--output_dir", dest='output_dir', default=None)


parser = argparse.ArgumentParser()
parser.set_defaults(function=encode_mcrae)
parser.add_argument("--n_train_concepts", dest='n_train_concepts', default=None)
parser.add_argument("--n_test_concepts", dest='n_test_concepts', default=None)
parser.add_argument("--output_train", dest='output_train')
parser.add_argument("--output_test", dest='output_test', default=None)

parser = argparse.ArgumentParser()
parser.set_defaults(function=encode_cslb)
parser.add_argument("--output_dir", dest='output_dir', default=None)