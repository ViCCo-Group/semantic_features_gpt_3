from skimage import io
import cv2
from matplotlib.pyplot import imshow, text
import statistics
from math import sqrt
from collections import defaultdict

colors = {
    'encyclopaedic': '#93C47D',
    'functional': '#FFD966',
    'conceptual': '#6D9EEB',
    'other perceptual': '#FF00FF',
    'visual': '#DD7E6B',
    'taxonomic': '#F6B26B' 
}

def plot(hist, ax, confs):
    print(hist)
    for value,count in hist.items():
        x = []
        y = []
        label = str(value)
        x.append(label)
        y.append(count)
        conf = confs[value]
        
        ax.bar(x, y, color=colors[label], yerr=[conf])
        #ymin, ymax = 0, 0.6
        #ax.set_ylim(ymin, ymax)

def get_confidence_interval(df):
    values = {
        'encyclopaedic': [],
        'functional': [],
        'conceptual': [], 
        'taxonomic': [],
        'visual': [],
        'other perceptual': []
    }
    confs = defaultdict()

    for i in range(1000):
        sample = df.sample(df.shape[0], replace=True)
        hist = sample['label'].value_counts(dropna=False, normalize=True)
        hist = hist.apply(lambda value: value * 100)
        for label in values:
            if label in hist:
                values[label].append(hist.loc[label])

    for label, label_values in values.items():
        stdev = statistics.stdev(label_values)
        confidence_interval = 1.96 * stdev 
        confs[label] = confidence_interval

    return confs
    

def plot_bar(df, ax, title):
    hist = df['label'].value_counts(dropna=False, normalize=True)
    hist = hist.apply(lambda value: value * 100)
    confs = get_confidence_interval(df)
    plot(hist, ax, confs)
    ax.set_title(title)
