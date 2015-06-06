'''
Created on Mar 6, 2015

@author: niko
'''

import numpy as np
import cPickle as pickle

from common import *

def initPredictionModel(bpf, modelFile=None, pretrainedFile=None):
    a = caffe.io.caffe_pb2.BlobProto();
    binaryProtoFile = open(bpf, 'rb')
    data = binaryProtoFile.read()
    a.ParseFromString(data)
    means=a.data
    means=np.asarray(means)
    means=means.reshape(3,256,256)
    
    # Set the right path to your model definition binaryProtoFile, pretrained model weights,
    # and the image you would like to classify.
    
    caffe.set_phase_test()
    #caffe.set_mode_cpu()
    caffe.set_mode_gpu()
    if modelFile is None:
        modelFile = MODEL_FILE
    if pretrainedFile is None:
        pretrainedFile = PRETRAINED
    net = caffe.Classifier(modelFile, pretrainedFile,
                           mean = means,
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(256, 256))
    
    return net

def getPrediction(model, fileName, getProbabilities=True):
    input_image = caffe.io.load_image(fileName)
    prediction = model.predict([input_image])
    if getProbabilities:
        return prediction[0]
    predictedClass = prediction[0].argmax()
    return predictedClass

def getActualValues(testFile):
    dict = {}
    itemsList = getTextEntriesFromFile(testFile)
    for item in itemsList:
        name, lbl = item.split(" ")
        n = name[:-5]
        dict[n] = int(lbl)
    return dict

def predictProbabilities(modelName, tag, sampleSubmissionFile, snapshot=100):
    if "sampleSubmission" in sampleSubmissionFile:
        probabilitiesPickleFile = DATA_PATH + '/submission_probabilities_' + tag + "_" + mdlName + '.p'
    else:
        probabilitiesPickleFile = DATA_PATH + '/validation_probabilities_' + tag + "_" + mdlName + '.p'
    try:
        probabilitiesDict = pickle.load(open(probabilitiesPickleFile, "rb"))
        processedItemsCount = len(probabilitiesDict)
        for k, v in probabilitiesDict.iteritems():
            vals = ",".join([str(val) for val in v])
            print "%s, %s" % (k, vals)
    except:
        probabilitiesDict = {}
        processedItemsCount = 0
        
    imagesList = getTextEntriesFromFile(sampleSubmissionFile)
    items = imagesList[1:]
    imagesCount = len(items)
    imagesProcessedCount = processedItemsCount
    start_time = time.time()
    time_previous_iter = start_time
    
    for item in items:
        itemLabel, itemClass = item.replace(",", " ").split()
        if itemLabel in probabilitiesDict.keys():
            continue
        itemFilename = "%s.jpeg" % itemLabel
        imageSourceFilename = "%s/%s" %(sourceImagesFolderTest, itemFilename)
        
        prediction = getPrediction(mdl, imageSourceFilename, getProbabilities=True)
        probabilitiesDict[itemLabel] = prediction
        
        imagesProcessedCount += 1        
        if imagesProcessedCount % snapshot == 0:
            elapsed_time = time.time() - start_time
            elapsed_time_previous_iter = time.time() - time_previous_iter
            time_previous_iter = time.time()
            secondsPerImage = elapsed_time_previous_iter/snapshot
            itemsRemaining = imagesCount - imagesProcessedCount
            secondsRemaining = secondsPerImage * itemsRemaining
            m, s = divmod(secondsRemaining, 60)
            h, m = divmod(m, 60)
            timeRemaining = "%d:%02d:%02d" % (h, m, s)
            pickle.dump(probabilitiesDict, open(probabilitiesPickleFile, 'wb'))
            print "Processed %d of %d items. Execution time: %.3f s (%f s/image; estimated remaining time: %s)" % (imagesProcessedCount, imagesCount, elapsed_time, secondsPerImage, timeRemaining)
        
    pickle.dump(probabilitiesDict, open(probabilitiesPickleFile, 'wb'))
    elapsed_time = time.time() - start_time
    m, s = divmod(elapsed_time, 60)
    h, m = divmod(m, 60)
    timeTotal = "%d:%02d:%02d" % (h, m, s)
    print "Prediction procedure for model %s finished. Probabilities stored to %s. Execution time: %s" % (modelDefinitions[i], probabilitiesPickleFile, timeTotal)

def generateSubmissionFileFromProbabilitiesDict(probabilitiesDict, resultsFile, sampleSubmissionFile=None, ordinal=False):
    storeItem("image,level", resultsFile)
    if sampleSubmissionFile is None:
        for k,v in probabilitiesDict.iteritems():
            predictedClass = v.argmax()
            maxProb = max(v)
            if ordinal:
                if maxProb <= 0.5:
                    predictedClass = 0
                    
            row = "%s,%d" % (k, predictedClass)
            storeItem(row, resultsFile)
    else:
        imagesList = getTextEntriesFromFile(sampleSubmissionFile)
        items = imagesList[1:]
        for item in items:
            itemLabel, itemClass = item.replace(",", " ").split()
            probabilities = probabilitiesDict[itemLabel]
            predictedClass = probabilities.argmax()
            maxProb = max(probabilities)
            if ordinal:
                if maxProb <= 0.5:
                    predictedClass = 0
            row = "%s,%d" % (itemLabel, predictedClass)
            storeItem(row, resultsFile)
        
    print "results stored to %s" % resultsFile
    
def getEnsemble(probabilitiesDictionariesList, sampleSubmissionFile, resultsFile):
    imagesList = getTextEntriesFromFile(sampleSubmissionFile)
    items = imagesList[1:]
    probabilities = {}
    storeItem("image,level", resultsFile)
    for item in items:
        itemLabel, itemClass = item.replace(",", " ").split()
        probs = [dict[itemLabel] for dict in probabilitiesDictionariesList]
        modelsCount = len(probs)
        classesCount = len(probs[0])
        votesForClasses = []
        for i in range(modelsCount):
            mdlProbs = probs[i]
            votes = [i[0] for i in sorted(enumerate(mdlProbs), key=lambda x: x[-1], reverse=False)]
            #classProbabilityAvg = sum(classProbabilities)*1.0/len(classProbabilities)
            #classProbabilityMax = max(classProbabilities)
            votesForClasses.append(votes)
        #probabilities = probabilitiesDict[itemLabel]
        votesSum = [ sum(x) for x in zip(*votesForClasses) ]
        predictedClass = np.asanyarray(votesSum).argmax()
        row = "%s,%d" % (itemLabel, predictedClass)
        storeItem(row, resultsFile)
        
    
if __name__ == "__main__":
        
    makePredictionsForSubmission = True
    configs = ['run-normal']
    #modelDefinitions = ['/home/niko/caffe-models/diabetic-retinopathy-detection/oxford_v1.prototxt']
    #pretrainedModels = ['/home/niko/caffe-models/diabetic-retinopathy-detection/snapshot/run-normal/small_kernels/oxford_v1_iter_450000.caffemodel']
    modelDefinitions = ['/home/niko/caffe-models/diabetic-retinopathy-detection/deep_v1_hdf5.prototxt']
    pretrainedModels = ['/home/niko/caffe-models/diabetic-retinopathy-detection/snapshot/run-normal/small_kernels/deep_v1_hdf5_iter_200000.caffemodel']
    for conf in configs:
        selectedFolder, sourceImagesFolderTrain, sourceImagesFolderTest, dataImagesTrain, dataImagesTest, trainLabelsFile, testLabelsFile, binaryProtoFile = getPathsForConfig(conf)
        for i in range(len(modelDefinitions)):
    
            mdl = initPredictionModel(binaryProtoFile, modelDefinitions[i], pretrainedModels[i])
            k = modelDefinitions[i].rfind("/")
            mdlName = modelDefinitions[i][k+1:-9]
            resultsFile = DATA_PATH + '/submission_' + conf + "_" + mdlName + '.csv'
            probabilitiesFile = DATA_PATH + '/probabilities_' + conf + "_" + mdlName + '.csv'
            probabilitiesPickleFile = DATA_PATH + '/probabilities_' + conf + "_" + mdlName + '.p'
            
            #predictProbabilities(mdlName, conf, SAMPLE_SUBMISSION_FILE)
            probabilitiesPickleFile = DATA_PATH + '/submission_probabilities_' + conf + "_" + mdlName + '.p'
            probabilitiesDict = pickle.load(open(probabilitiesPickleFile, "rb"))
            generateSubmissionFileFromProbabilitiesDict(probabilitiesDict, resultsFile, SAMPLE_SUBMISSION_FILE, ordinal=True)
            exit
            
            
                
                
    
    