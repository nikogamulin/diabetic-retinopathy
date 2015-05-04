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

CONFIG = 'run-contrast-1'

DATA_PATH = "/home/niko/datasets/DiabeticRetinopathyDetection"
MODEL_PATH = '/home/niko/caffe-models/diabetic-retinopathy-detection'
VALIDATION_PATH = '/home/niko/caffe-models/diabetic-retinopathy-detection/validation'
VALIDATION_FILE = VALIDATION_PATH + '/test.txt'

'''
SELECTED_FOLDER = DATA_PATH + "/processed/" + CONFIG

SOURCE_IMAGES_FOLDER_TRAIN = SELECTED_FOLDER + '/train'
SOURCE_IMAGES_FOLDER_TEST = SELECTED_FOLDER + '/test'

DATA_IMAGES_TRAIN = SELECTED_FOLDER + '/train_train'
DATA_IMAGES_TEST = SELECTED_FOLDER + '/train_test'

DATA_IMAGES_TEST_AUGMENTED = SELECTED_FOLDER + '/train_test_augmented'

TRAIN_LABELS_FILE = SELECTED_FOLDER + "/training.txt"
TEST_LABELS_FILE = SELECTED_FOLDER + "/test.txt"
'''

LABELS = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

MODEL_FILE = '/home/niko/caffe-models/diabetic-retinopathy-detection/lenet.prototxt'
#MODEL_FILE = '/home/niko/caffe-models/diabetic-retinopathy-detection/lenet_7_7_dropout.prototxt'
#PRETRAINED = '/home/niko/caffe-models/diabetic-retinopathy-detection/finetune_diabetic_retinopathy_256_iter_60000.caffemodel'
PRETRAINED = '/home/niko/caffe-models/diabetic-retinopathy-detection/snapshot/run-normal/lenet_normal_iter_20000.caffemodel'
#PRETRAINED = '/home/niko/caffe-models/diabetic-retinopathy-detection/snapshot/run_normal_7_7/lenet_normal_iter_20000.caffemodel'
#PRETRAINED = '/home/niko/caffe-models/diabetic-retinopathy-detection/snapshot/run_normal_7_7_dropout/lenet_normal_iter_50000.caffemodel'
IMAGE_FILES = ['/home/niko/datasets/DiabeticRetinopathyDetection/processed/run-normal/test/1_left.jpeg', '/home/niko/datasets/DiabeticRetinopathyDetection/processed/run-normal/test/9_left.jpeg']

#BINARY_PROTO_FILE = '/home/niko/datasets/DiabeticRetinopathyDetection/augmented/diabetic_retinopathy_mean_256_256.binaryproto'
#BINARY_PROTO_FILE = '/home/niko/datasets/DiabeticRetinopathyDetection/processed/run-normal/diabetic_retinopathy_mean.binaryproto'

TRAIN_LABELS_FILE_SOURCE = "%s/trainLabels.csv" % DATA_PATH
TRAIN_AUGMENTED_LABELS_FILE_SOURCE = "%s/trainLabelsAugmented.csv" % DATA_PATH
SAMPLE_SUBMISSION_FILE = "%s/sampleSubmission.csv" % DATA_PATH

def getPathsForConfig(conf):

    SELECTED_FOLDER = DATA_PATH + "/processed/" + conf    
    SOURCE_IMAGES_FOLDER_TRAIN = SELECTED_FOLDER + '/train'
    SOURCE_IMAGES_FOLDER_TEST = SELECTED_FOLDER + '/test'
    DATA_IMAGES_TRAIN = SELECTED_FOLDER + '/train_train'
    DATA_IMAGES_TEST = SELECTED_FOLDER + '/train_test' 
    DATA_IMAGES_TEST_AUGMENTED = SELECTED_FOLDER + '/train_test_augmented'
    TRAIN_LABELS_FILE = SELECTED_FOLDER + "/training.txt"
    TEST_LABELS_FILE = SELECTED_FOLDER + "/test.txt"
    #BINARY_PROTO_FILE = '/home/niko/datasets/DiabeticRetinopathyDetection/processed/run-normal/diabetic_retinopathy_mean.binaryproto'
    BINARY_PROTO_FILE = SELECTED_FOLDER + '/diabetic_retinopathy_mean.binaryproto'
    
    return [SELECTED_FOLDER, SOURCE_IMAGES_FOLDER_TRAIN, SOURCE_IMAGES_FOLDER_TEST, DATA_IMAGES_TRAIN, DATA_IMAGES_TEST, DATA_IMAGES_TEST_AUGMENTED, TRAIN_LABELS_FILE, TEST_LABELS_FILE, BINARY_PROTO_FILE]

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
