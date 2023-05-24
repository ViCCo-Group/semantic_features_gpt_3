import sys 
import os 

sys.path.append('../..')

FIGURES_DIR = '../../figures'
from os.path import join as pjoin

import matplotlib.pyplot as plt 
import pickle
from utils.analyses.category.pairiwise import plot_violin

tex_fonts = {
        # Use LaTeX to write all text
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8
}
plt.rcParams.update(tex_fonts)

with open('pairwirse.pkl', 'rb') as results:
    data = pickle.load(results)

fig, axes = plt.subplots(1,1, sharex=True)
plot_violin(axes, data['gpt'], data['cslb'], data['mc'], data['categories'])
fig.set_size_inches(7, 2)
axes.legend(bbox_to_anchor=(1.1, 1))
plt.savefig(pjoin(FIGURES_DIR, 'pairwise_similarities.svg'), bbox_inches='tight')
