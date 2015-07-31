'''
Created on Mar 8, 2015

@author: niko
'''

import numpy as np
import matplotlib.pyplot as plt


def getDatasetStats():
    hist, bin_edges = np.histogram([1, 1, 2, 2, 2, 2, 3], bins = range(5))
    plt.bar(bin_edges[:-1], hist, width = 1)
    plt.xlim(min(bin_edges), max(bin_edges))
    plt.show()
