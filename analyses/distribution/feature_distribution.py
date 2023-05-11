import sys 
import os 

sys.path.append('../..')
DATA_DIR = '../../data'
os.environ['DATA_DIR'] = DATA_DIR
FIGURE_DIR = '../../figures'

from utils.analyses.distribution.distribution import plot_bar

import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from os.path import join as pjoin 


gpt_mcrae_label_df = pd.read_csv(f'{DATA_DIR}/feature_labels/gpt_mcrae.csv')
gpt_cslb_label_df = pd.read_csv(f'{DATA_DIR}/feature_labels/gpt_cslb.csv')
cslb_label_df = pd.read_csv(f'{DATA_DIR}/feature_labels/cslb.csv')
mc_label_df = pd.read_csv(f'{DATA_DIR}/feature_labels/mc.csv')

fig, axes = plt.subplots(2,2,figsize=(30,20), sharey=True, sharex=True)
plot_bar(cslb_label_df, axes[0][1], 'CSLB (human generated)')
plot_bar(gpt_cslb_label_df, axes[1][1], 'GPT-3 primed with CSLB (our approach)')

plot_bar(mc_label_df, axes[0][0], 'McRae (human generated)')
plot_bar(gpt_mcrae_label_df, axes[1][0], 'GPT-3 primed with McRae (our approach)')

axes[1][0].set_ylabel("Relative number of features in percentage", fontsize=25)
plt.tight_layout()
fig.savefig(pjoin(FIGURE_DIR, 'distribution_of_labels.svg'))
