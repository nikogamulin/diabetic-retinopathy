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
    itemsCount = sum(stats)
    classesCount = len(stats)
    n_dups = [int(round((1.0/classesCount) / (float(classItemsCount) / itemsCount) )) for classItemsCount in stats]
    return n_dups


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
    #hist_equalized = exposure.equalize_hist(hsv)
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
            
def getAugmentedImages(img, name, rotationsCount):
    augmentedImages = []
    augmentedImagesLabels = []
    basicAugmentations = getBasicAugmentations(img)
    pcaAugmentations = getPCAAugmentations(basicAugmentations)
    for j in range(len(pcaAugmentations)):
        img = pcaAugmentations[j]
        rotationAngles = [random.randint(0,360) for p in range(rotationsCount)]
        rotationAngles.append(0)
        for orientation in rotationAngles:
            imName = "%s_%d_%d" % (name,j, orientation)
            imr = transform.rotate(img, angle = orientation)
            augmentedImages.append(imr)
            augmentedImagesLabels.append(imName)
    return [augmentedImages, augmentedImagesLabels]

def saveAugmentedImages(images, folder, labelsFile, label):
    for i in range(len(images[0])):
        imName = images[1][i]
        img = images[0][i]
        itemToStore = "%s.jpeg %d" % (imName, label)
        storeItem(itemToStore, labelsFile)
        fname = "%s/%s.jpeg" % (folder, imName)
        io.imsave(fname, img)
            
        
    
def splitDatasetToTrainingAndTestDataset(balancedDataset = True):
    imagesCount, items = getItemsFromFile(filename = TRAIN_AUGMENTED_LABELS_FILE_SOURCE, excludeHeader = True)
    itemsDict = {}
    for item in items:
        name, lbl = item.split(",")
        itemsDict[name] = int(lbl)
        
    stats = getDatasetStats()
    indices = [re.findall(r'\d+', i) for i in items]
    uniqueIndices = list(set([int(i[0]) for i in indices]))
    shuffle(uniqueIndices)
    imagesProcessedCount = 0
    start_time = time.time()
    statsTest = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    labels = [PRETRAIN_LABELS_FILE, TRAIN_LABELS_FILE, TEST_LABELS_FILE]
    for index in uniqueIndices:
        #try to get left and right eye images labels
        imageName = "%d_left" % index
        if imageName in itemsDict:
            leftEyeLabel = itemsDict[imageName]
            destination = determineTargetLabelsAndDataset(leftEyeLabel, stats, balancedDataset)
            itemFilename = "%s.jpeg" % imageName
            itemSourceFilename = "%s/%s" %(AUGMENTED_IMAGES_FOLDER, itemFilename)
            itemToStore = "%s %d" % (itemFilename, leftEyeLabel)
            storeItem(itemToStore, destination[0])
            shutil.copy2(itemSourceFilename, destination[1])
            idx = labels.index(destination[0])
            imagesProcessedCount += 1
            statsTest[leftEyeLabel][idx] += 1
        else:
            destination = None
        imageName = "%d_right" % index
        if imageName in itemsDict:
            idx = labels.index(destination[0])
            rightEyeLabel = itemsDict[imageName]
            if destination is None:
                destination = determineTargetLabelsAndDataset(rightEyeLabel, stats, balancedDataset)
                idx = labels.index(destination[0])
            itemFilename = "%s.jpeg" % imageName
            itemSourceFilename = "%s/%s" %(SOURCE_IMAGES_FOLDER, itemFilename)
            itemToStore = "%s %d" % (itemFilename, rightEyeLabel)
            storeItem(itemToStore, destination[0])
            shutil.copy2(itemSourceFilename, destination[1])
            statsTest[rightEyeLabel][idx] += 1
            imagesProcessedCount += 1
            
        if imagesProcessedCount % 10000 == 0:
            print statsTest
            elapsed_time = time.time() - start_time
            print "Processed %d of %d items. Execution time: %.3f s" % (imagesProcessedCount, imagesCount, elapsed_time)
            
def splitDatasetToTrainingAndTestDataset2(SOURCE_IMAGES_FOLDER_TRAIN, folderImagesTrain, folderImagesTest, trainLabelsFile, testLabelsFile, augmentTestSet=False):
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
                itemSourceFilename = "%s/%s" %(SOURCE_IMAGES_FOLDER_TRAIN, itemFilename)
                normalizedImage = io.imread(itemSourceFilename)
                if destination[1] == folderImagesTrain:
                    images = getAugmentedImages(normalizedImage, imageName, duplicatesCount[eyeLabel])
                    saveAugmentedImages(images, folderImagesTrain, destination[0], eyeLabel)
                else:
                    if augmentTestSet:
                        images = getAugmentedImages(normalizedImage, imageName, duplicatesCount[eyeLabel])
                        saveAugmentedImages(images, folderImagesTest, destination[0], eyeLabel)
                    else:
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
    
    
    
if __name__ == "__main__":
    augmentTestSet = False
    configurations = ['run-normal']
    for conf in configurations:
        selectedFolder, sourceImagesFolderTrain, sourceImagesFolderTest, folderImagesTrain, folderImagesTest, FolderImagesTestAugmented, trainLabelsFile, testLabelsFile, binaryProtoFile = getPathsForConfig(conf)
    
        #folderImagesTrain = DATA_IMAGES_TRAIN
        if augmentTestSet:
            folderImagesTest = FolderImagesTestAugmented

        
        #trainLabelsFile = TRAIN_LABELS_FILE
        #testLabelsFile = TEST_LABELS_FILE
        
        os.chdir("/usr/local/caffe")
        
        
        if PREPARE_IMAGES:
            
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
          
            #splitDatasetToTrainingAndTestDataset2(augmentTestSet=augmentTestSet)
            splitDatasetToTrainingAndTestDataset2(sourceImagesFolderTrain, folderImagesTrain, folderImagesTest, trainLabelsFile, testLabelsFile, augmentTestSet=augmentTestSet) 
            
            print "Images split."
        
              
        cmdTrainLmdb = "GLOG_logtostderr=1 build/tools/convert_imageset \
            --resize_height=256 \
            --resize_width=256 \
            --shuffle \
            %s/ \
            %s \
            %s/diabetic_retinopathy_train_lmdb" % (folderImagesTrain, trainLabelsFile, selectedFolder)
            
        cmdValLmdb = "GLOG_logtostderr=1 build/tools/convert_imageset \
            --shuffle \
            %s/ \
            %s \
            %s/diabetic_retinopathy_val_lmdb" % (folderImagesTest, testLabelsFile, selectedFolder)
            
        cmdCreateImagenetMean = "./build/tools/compute_image_mean %s/diabetic_retinopathy_train_lmdb \
      %s" % (selectedFolder, binaryProtoFile)
            
        print "Creating train lmdb..."
        return_code = subprocess.call(cmdTrainLmdb, shell=True)
        
        print "Creating val lmdb..."        
        
        return_code = subprocess.call(cmdValLmdb, shell=True)
        
        print "Done."
        
        print "Creating binary proto file..."  
        return_code = subprocess.call(cmdCreateImagenetMean, shell=True)
        
        print "finished processing images for %s" % conf



