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

tex_fonts = {
        # Use LaTeX to write all text
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 10,
        "xtick.labelsize": 4,
        "ytick.labelsize": 8
}
plt.rcParams.update(tex_fonts)

gpt_mcrae_label_df = pd.read_csv(f'{DATA_DIR}/feature_labels/gpt_mcrae.csv')
gpt_cslb_label_df = pd.read_csv(f'{DATA_DIR}/feature_labels/gpt_cslb.csv')
cslb_label_df = pd.read_csv(f'{DATA_DIR}/feature_labels/cslb.csv')
mc_label_df = pd.read_csv(f'{DATA_DIR}/feature_labels/mc.csv')

fig, axes = plt.subplots(2,2,figsize=(7,5), sharey=True, sharex=True)
plot_bar(cslb_label_df, axes[0][1], 'CSLB (human generated)')
plot_bar(gpt_cslb_label_df, axes[1][1], 'GPT-3 primed with CSLB (our approach)')

plot_bar(mc_label_df, axes[0][0], 'McRae (human generated)')
plot_bar(gpt_mcrae_label_df, axes[1][0], 'GPT-3 primed with McRae (our approach)')

axes[1][0].set_ylabel("Relative number of features in %")
axes[1][0].set_xlabel('Feature label')
axes[1][1].set_xlabel('Feature label')

fig.savefig(pjoin(FIGURE_DIR, 'distribution_of_labels.svg'), bbox_inches="tight")
