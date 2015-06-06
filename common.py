'''
Created on Mar 8, 2015

@author: niko
'''

import Image
import ImageTk
import os
import glob
import random
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import re
import os
import sys

caffe_root = '/usr/local/caffe/'  # this file is expected to be in {caffe_root}/examples

sys.path.insert(0, caffe_root + 'python')

import caffe

DATA_PATH = "/home/niko/datasets/DiabeticRetinopathyDetection"
MODEL_PATH = '/home/niko/caffe-models/diabetic-retinopathy-detection'
VALIDATION_PATH = '/home/niko/caffe-models/diabetic-retinopathy-detection/validation'
VALIDATION_FILE = VALIDATION_PATH + '/test.txt'

LABELS = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

MODEL_FILE = '/home/niko/caffe-models/diabetic-retinopathy-detection/lenet.prototxt'

PRETRAINED = '/home/niko/caffe-models/diabetic-retinopathy-detection/snapshot/run-normal/lenet_normal_iter_20000.caffemodel'

IMAGE_FILES = ['/home/niko/datasets/DiabeticRetinopathyDetection/processed/run-normal/test/1_left.jpeg', '/home/niko/datasets/DiabeticRetinopathyDetection/processed/run-normal/test/9_left.jpeg']


TRAIN_LABELS_FILE_SOURCE = "%s/trainLabels.csv" % DATA_PATH
TRAIN_AUGMENTED_LABELS_FILE_SOURCE = "%s/trainLabelsAugmented.csv" % DATA_PATH
SAMPLE_SUBMISSION_FILE = "%s/sampleSubmission.csv" % DATA_PATH

def getPathsForConfig(conf, destinationFolder=None):

    sourceFolder = DATA_PATH + "/processed/" + conf    
    sourceImagesFolderTrain = sourceFolder + '/train'
    sourceImagesFolderTest = sourceFolder + '/test'
    if destinationFolder is None:
        destinationFolderImagesTrain = sourceFolder + '/train_train'
        destinationFolderImagesTest = sourceFolder + '/train_test' 
        trainLabelsFile = sourceFolder + "/labelsTrain.txt"
        testLabelsFile = sourceFolder + "/labelsTest.txt"
        binaryProtoFile = sourceFolder + '/diabetic_retinopathy_mean.binaryproto'
        destinationFolder = sourceFolder
    else:
        destinationFolderImagesTrain = destinationFolder + '/train'
        destinationFolderImagesTest = destinationFolder + '/test' 
        trainLabelsFile = destinationFolder + "/labelsTrain.txt"
        testLabelsFile = destinationFolder + "/labelsTest.txt"
        binaryProtoFile = destinationFolder + '/diabetic_retinopathy_mean.binaryproto'
    return [destinationFolder, sourceImagesFolderTrain, sourceImagesFolderTest, destinationFolderImagesTrain, destinationFolderImagesTest, trainLabelsFile, testLabelsFile, binaryProtoFile]

def getItemsFromFile(filename = TRAIN_LABELS_FILE_SOURCE, excludeHeader = True):
    imagesList = getTextEntriesFromFile(filename)
    imagesCount = len(imagesList) - 1
    if excludeHeader:
        imagesList = imagesList[1:]
    return [imagesCount, imagesList]

def getTextEntriesFromFile(fname):
    with open(fname) as f:
        content = [x.strip('\n') for x in f.readlines()]
        return content

def storeImgLbl(lbl, imgPath, destFile):
    with open(destFile, 'a+') as f:
        f.write(imgPath + ' ' + str(lbl) + '\n')
        
def storeItem(itemValue, destFile):
    with open(destFile, 'a+') as f:
        f.write(itemValue + '\n')
        
def storeItemOrdinal(itemValue, destFile, categoriesCount):
    filename, category = itemValue.split(" ")
    outputVectorLength = categoriesCount
    outputVector = [0] * outputVectorLength
    for i in range(outputVectorLength):
        if i < int(category):
            outputVector[i] = 1
        else:
            outputVector[i] = -1
    strOutputVector = " ".join(str(x) for x in outputVector)
    itemValueOrdinal ="%s %s" % (filename, strOutputVector)
    with open(destFile, 'a+') as f:
        f.write(itemValueOrdinal + '\n')
        
def recodeCategoricalToOrdinal(fileSource, fileDestination, categoriesCount):
    itemscount,items = getItemsFromFile(filename=fileSource,excludeHeader=False)
    for item in items:
        storeItemOrdinal(item, fileDestination, categoriesCount)
        
    
        
def getDatasetStats(filename = TRAIN_LABELS_FILE_SOURCE, showHistogram = False):
    classes = []
    itemscount,items = getItemsFromFile(filename=filename)
    for item in items:
        #index, lbl = re.findall(r'\d+', item)
        if ',' in item:
            index, lbl = item.split(",")
        else:
            index, lbl = item.split(" ")
        classes.append(int(lbl))
    hist, bin_edges = np.histogram(classes, bins = range(len(LABELS)+1))
    if showHistogram:
        plt.bar(bin_edges[:-1], hist, width = 1)
        plt.xlim(min(bin_edges), max(bin_edges))
        plt.show()
    return hist.tolist()
