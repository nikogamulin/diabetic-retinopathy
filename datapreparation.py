'''
Created on Mar 5, 2015

@author: niko
'''
from skimage import io, transform, color, exposure
import numpy as np
import pylab
import matplotlib.pyplot as plt
import time
import random
import image_pca
import image_affine
import string
import copy

'''
Created on Feb 10, 2015

@author: niko
'''

from common import *

TRAINING_SET_SIZE = 0.7
VISUALIZE_TRANSFORMATIONS = False
PREPARE_IMAGES = True

def getNumberOfDuplicates():
    stats = getDatasetStats()
    cumulativeStats = np.cumsum(stats)
    itemsCount = sum(stats)
    classesCount = len(stats)
    n_dups = [int(round((1.0/classesCount) / (float(classItemsCount) / itemsCount) )) for classItemsCount in stats]
        
    return n_dups

def addCategoryTag(name, category):
    if '.' in name:
        result = string.replace(name, '.', '_category_%d.' % category)
    else:
        result = "%s_category_%d" % (name, category)
    return result


def getNormalizedImage(filename, cropRectangle = True, scale=256):
    img = io.imread(filename)
    height, width, channels = img.shape
    if cropRectangle:
        if width > height:
            delta = width - height
            left = int(delta/2)
            upper = 0
            right = height + left
            lower = height
        else:
            delta = height - width
            left = 0
            upper = int(delta/2)
            right = width
            lower = width + upper
        img = img[upper:lower, left:right]
        if not scale is None:
            img = transform.resize(img, (scale,scale))
    hsv = color.rgb2hsv(img)
    h = np.copy(hsv[:,:,0])
    s = np.copy(hsv[:,:,1])
    v = np.copy(hsv[:,:,2])
    hist_equalized_h = exposure.equalize_hist(h)
    hist_equalized_s = exposure.equalize_hist(s)
    hist_equalized_v = exposure.equalize_hist(v)
    hist_equalized = np.copy(hsv)
    hist_equalized[:,:,0] = hist_equalized_h
    hist_equalized[:,:,1] = hist_equalized_s
    hist_equalized[:,:,2] = hist_equalized_v
    if VISUALIZE_TRANSFORMATIONS:
        fig, axes = pylab.subplots(nrows = 4, ncols = 1, figsize = (4, 4 * 3))
        pylab.gray()
        axes[0].imshow(img[:,:,0])
        axes[0].set_title('unequalized image')
        axes[1].imshow(hist_equalized[:,:,0])
        axes[1].set_title('hist-equalized image (R)')
        axes[2].imshow(hist_equalized[:,:,1])
        axes[2].set_title('hist-equalized image (G)')
        axes[3].imshow(hist_equalized[:,:,2])
        axes[3].set_title('hist-equalized image (B)')
        plt.show()
    return hist_equalized

def getBasicAugmentations(img):
    imr = transform.rotate(img, angle = 45)
    imgFlipped = np.fliplr(img)
    imrFlippedRotated = np.fliplr(imr)
    if VISUALIZE_TRANSFORMATIONS:
        fig, axes = pylab.subplots(nrows = 4, ncols = 1, figsize = (4, 4 * 3))
        pylab.gray()
        axes[0].imshow(img)
        axes[0].set_title('original image')
        axes[1].imshow(imr)
        axes[1].set_title('rotated image')
        axes[2].imshow(imgFlipped)
        axes[2].set_title('flipped image')
        axes[3].imshow(imrFlippedRotated)
        axes[3].set_title('flipped rotated image')
        plt.show()
    return [img, imr, imgFlipped, imrFlippedRotated]

def getPCAAugmentations(images):
    result = []
    for img in images:
        result.append(img)
        try:
            pcaAugmented = image_pca.distortImage(img, 3)
            result.append(pcaAugmented)
        except:
            pass
    return result

def getShearAugmentations(images):
    result = []
    for img in images:
        result.append(img)
        try:           
            modified = image_affine.getShearedImage(img)
            result.append(modified)
        except:
            pass
    return result
        
def getFlippedImage(img):
    flipped_ud = np.flipud(img)
    if VISUALIZE_TRANSFORMATIONS:
        fig, axes = pylab.subplots(nrows = 3, ncols = 1, figsize = (4, 4 * 2))
        pylab.gray()
        axes[0].imshow(img)
        axes[0].set_title('original image')
        axes[1].imshow(flipped_ud)
        axes[1].set_title('flipped image')
        plt.show()
    return flipped_ud

def determineTargetLabelsAndDataset(trainLabelsFile, testLabelsFile, folderImagesTrain, folderImagesTest):
    labels = [trainLabelsFile, testLabelsFile]
    folders = [folderImagesTrain, folderImagesTest]
    rndVal = random.random()
    if rndVal < TRAINING_SET_SIZE:
        return [labels[0], folders[0]]
    else:
        return [labels[1], folders[1]]
            
def getAugmentedImages(img, name, rotationsCount, doShearAugmentations=False, doPcaAugmentations=False):
    augmentedImages = []
    augmentedImagesLabels = []
    basicAugmentations = getBasicAugmentations(img)
    if doPcaAugmentations:
        basicAugmentations = getPCAAugmentations(basicAugmentations)
    if doShearAugmentations:
        basicAugmentations = getShearAugmentations(basicAugmentations)
    for j in range(len(basicAugmentations)):
        img = basicAugmentations[j]
        rotationAngles = [random.randint(0,360) for p in range(rotationsCount)]
        rotationAngles.append(0)
        for orientation in rotationAngles:
            imName = "%s_%d_%d" % (name,j, orientation)
            imr = transform.rotate(img, angle = orientation)
            augmentedImages.append(imr)
            augmentedImagesLabels.append(imName)
    return [augmentedImages, augmentedImagesLabels]

def saveAugmentedImages(images, folder, labelsFile, label, configTag=None):
    for i in range(len(images[0])):
        imName = images[1][i]
        if not configTag is None:
            imName = configTag + '_' + imName
        img = images[0][i]
        itemToStore = "%s.jpeg %d" % (imName, label)
        storeItem(itemToStore, labelsFile)
        fname = "%s/%s.jpeg" % (folder, imName)
        io.imsave(fname, img)      
            
def splitDatasetToTrainingAndTestDataset(configTag, sourceImagesFolderTrain, folderImagesTrain, folderImagesTest, trainLabelsFile, testLabelsFile):
    duplicatesCount = getNumberOfDuplicates()
    imagesCount, items = getItemsFromFile(filename = TRAIN_LABELS_FILE_SOURCE, excludeHeader = True)
    itemsDict = {}
    for item in items:
        name, lbl = item.split(",")
            
        itemsDict[name] = int(lbl)
        
    indices = [re.findall(r'\d+', i) for i in items]
    uniqueIndices = list(set([int(i[0]) for i in indices]))
    imagesProcessedCount = 0
    start_time = time.time()
    sides = ['left', 'right']
    for index in uniqueIndices:
        #try to get left and right eye images labels
        destination = None
        for side in sides:
            imageName = "%d_%s" % (index, side)
            if imageName in itemsDict:
                eyeLabel = itemsDict[imageName]
                if destination is None:
                    destination = determineTargetLabelsAndDataset(trainLabelsFile, testLabelsFile, folderImagesTrain, folderImagesTest)
                itemFilename = "%s.jpeg" % imageName
                itemSourceFilename = "%s/%s" %(sourceImagesFolderTrain, itemFilename)
                normalizedImage = io.imread(itemSourceFilename)
                if destination[1] == folderImagesTrain:
                    images = getAugmentedImages(normalizedImage, imageName, duplicatesCount[eyeLabel], doShearAugmentations=True, doPcaAugmentations=True)
                    saveAugmentedImages(images, folderImagesTrain, destination[0], eyeLabel, configTag)
                else:
                    imageName = configTag + '_' + imageName
                    itemToStore = "%s.jpeg %d" % (imageName, eyeLabel)
                    storeItem(itemToStore, destination[0])
                    fname = "%s/%s.jpeg" % (folderImagesTest, imageName)
                    io.imsave(fname, normalizedImage)
            imagesProcessedCount += 1            
            if imagesProcessedCount % 10000 == 0:
                elapsed_time = time.time() - start_time
                print "Processed %d of %d items. Execution time: %.3f s" % (imagesProcessedCount, imagesCount, elapsed_time)
                
    elapsed_time = time.time() - start_time
    print "Processed %d of %d items. Execution time: %.3f s" % (imagesProcessedCount, imagesCount, elapsed_time) 

def prepareDatasetsForSelectedConfiguration(configurations, prepareImages=False, clearExisting=False, create_lmdb=False, ordinalEncoding=True, commonDestination=None):
    for config in configurations:
        c = getPathsForConfig(config, commonDestination)
        destinationFolder, sourceImagesFolderTrain, sourceImagesFolderTest, folderImagesTrain, folderImagesTest, trainLabelsFile, testLabelsFile, binaryProtoFile = c
        
        os.chdir("/usr/local/caffe")
        
        if clearExisting:
            
            if os.path.exists(folderImagesTrain):
                shutil.rmtree(folderImagesTrain)
            os.makedirs(folderImagesTrain)
            if os.path.exists(folderImagesTest):
                shutil.rmtree(folderImagesTest)
            os.makedirs(folderImagesTest)
            
            try:
                os.remove(trainLabelsFile)
            except OSError:
                pass
            
            try:
                os.remove(testLabelsFile)
            except OSError:
                pass     
        if prepareImages: 
            
            if not os.path.exists(folderImagesTrain):
                os.makedirs(folderImagesTrain)
            if not os.path.exists(folderImagesTest):
                os.makedirs(folderImagesTest)
                    
            splitDatasetToTrainingAndTestDataset(config, sourceImagesFolderTrain, folderImagesTrain, folderImagesTest, trainLabelsFile, testLabelsFile) 
            
            print "Images split for configuration %s" % config
            
    if create_lmdb:
        
        lblTrainLmdb = "diabetic_retinopathy_train_lmdb"
        lblTestLmdb = "diabetic_retinopathy_test_lmdb"
        
        if ordinalEncoding:
            trainLabelsFileRecoded = trainLabelsFile.replace(".", "_ordinal.")
            testLabelsFileRecoded = testLabelsFile.replace(".", "_ordinal.")
            if not os.path.isfile(trainLabelsFileRecoded):
                recodeCategoricalToOrdinal(trainLabelsFile, trainLabelsFileRecoded, 5)
            if not os.path.isfile(testLabelsFileRecoded):
                recodeCategoricalToOrdinal(testLabelsFile, testLabelsFileRecoded, 5)
            testLabelsFile = testLabelsFileRecoded
            trainLabelsFile = trainLabelsFileRecoded
            
            lblTrainLmdb = "diabetic_retinopathy_train_ordinal.leveldb"
            lblTestLmdb = "diabetic_retinopathy_test_ordinal.leveldb"
        
              
        cmdTrainLmdb = "GLOG_logtostderr=1 build/tools/convert_imageset \
            --resize_height=256 \
            --resize_width=256 \
            --shuffle \
            %s/ \
            %s \
            %s/%s" % (folderImagesTrain, trainLabelsFile, destinationFolder, lblTrainLmdb)
            
        cmdValLmdb = "GLOG_logtostderr=1 build/tools/convert_imageset \
            --resize_height=256 \
            --resize_width=256 \
            --shuffle \
            %s/ \
            %s \
            %s/%s" % (folderImagesTest, testLabelsFile, destinationFolder, lblTestLmdb)
            
        cmdCreateImagenetMean = "./build/tools/compute_image_mean %s/%s \
      %s" % (destinationFolder, lblTrainLmdb, binaryProtoFile)
            
        print "Creating train lmdb..."
        return_code = subprocess.call(cmdTrainLmdb, shell=True)
        
        print "Creating val lmdb..."        
        
        return_code = subprocess.call(cmdValLmdb, shell=True)
        
        print "Done."
        
        print "Creating binary proto file..."  
        return_code = subprocess.call(cmdCreateImagenetMean, shell=True)
        
        notification = "finished processing images."
    
    print notification
    
if __name__ == "__main__":
    
    labelsFolder = '/home/niko/datasets/DiabeticRetinopathyDetection/processed/run-normal'
    destinationFolder = '/media/niko/data/data/DiabeticRetinopathy'
    configurations = ['run-normal', 'run-contrast-1', 'run-contrast-2', 'run-hue-1', 'run-hue-2', 'run-sat-1', 'run-sat-2', 'run-stretch']
    #(configurations, prepareImages=False, clearExisting=False, ordinalEncoding=True, commonDestination=None)
    prepareDatasetsForSelectedConfiguration(configurations, prepareImages=False, ordinalEncoding=False, create_lmdb=True, commonDestination=destinationFolder)
        



