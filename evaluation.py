'''
Created on Mar 26, 2015

@author: niko
'''

import common
from quadratic_weighted_kappa import *

def calculateQuadraticWeightedKappa(s):
    lines = common.getTextEntriesFromFile(s)
    valuesActual = []
    valuesPredicted = []
    for line in lines:
        items = line.split(" ")
        valuesActual.append(int(items[1]))
        valuesPredicted.append(int(items[2]))
    qwk = quadratic_weighted_kappa(valuesActual, valuesPredicted, min_rating=0, max_rating=4)
    return qwk

if __name__ == '__main__':
    quadraticWeightedKappa = calculateQuadraticWeightedKappa('/home/niko/caffe-models/diabetic-retinopathy-detection/validation/lenet.txt')
    print 'quadratic weighted kappa = %f' % quadraticWeightedKappa