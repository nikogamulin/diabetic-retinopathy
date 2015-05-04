'''
Created on Mar 12, 2015

@author: niko
'''

import logparser
from pylab import *

def plotLoss(trainNumIters, trainingLoss, testNumIters, testLoss, trainLearningRate, testAccuracy):
    fig, ax1 = plt.subplots()
    pltTrainingError = ax1.plot(trainNumIters, trainingLoss, 'r', label = 'Training')
    pltTestError = ax1.plot(testNumIters, testLoss, 'b', label = 'Test')
    ax1.set_ylabel(r"Loss", color="black")
    ax1.set_xlim([0, max(testNumIters)])
        
    '''
    ax2 = ax1.twinx()
    pltLearningRate = ax2.plot(trainNumIters, trainLearningRate, label="Learning rate", lw=1, color="black")
    ax2.set_ylabel(r"Learning rate $(\alpha)$", color="black")
    ax1.set_xlim([0, max(testNumIters)])
    ax2.set_yscale('log')
    '''
    
    ax2 = ax1.twinx()
    pltLearningRate = ax2.plot(testNumIters, testAccuracy, label="Accuracy", lw=1, color="black")
    ax2.set_ylabel(r"Accuracy", color="black")
    ax1.set_xlim([0, max(testNumIters)])
    
    lns = pltTrainingError+pltTestError+pltLearningRate
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)

    show()
    
def plotAccuracy(trainNumIters, trainingLoss, testNumIters, testAccuracy):
    fig, ax1 = plt.subplots()
    pltTrainingError = ax1.plot(trainNumIters, trainingLoss, 'r', label = 'Training Loss')
    ax1.set_ylabel(r"Loss", color="black")
    ax1.set_xlim([0, max(testNumIters)])
        
    ax2 = ax1.twinx()
    pltTestAccuracy = ax2.plot(testNumIters, testAccuracy, label="Test Accuracy", lw=1, color="black")
    ax2.set_ylabel(r"Accuracy", color="black")
    ax1.set_xlim([0, max(testNumIters)])
    
    lns = pltTrainingError+pltTestAccuracy
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)

    show()

if __name__ == '__main__':

    #train_dict_list, train_dict_names, test_dict_list, test_dict_names = logparser.parse_log("/home/niko/caffe-models/diabetic-retinopathy-detection/512/454/log_512_overfitting.txt")
    #train_dict_list, train_dict_names, test_dict_list, test_dict_names = logparser.parse_log("/home/niko/caffe-models/diabetic-retinopathy-detection/log/log_run_normal_7_7.txt")
    #train_dict_list, train_dict_names, test_dict_list, test_dict_names = logparser.parse_log("/home/niko/caffe-models/diabetic-retinopathy-detection/log/log_run_contrast_1.txt")
    #train_dict_list, train_dict_names, test_dict_list, test_dict_names = logparser.parse_log("/home/niko/caffe-models/diabetic-retinopathy-detection/log/log_run_normal_7_7_dropout.txt")
    #najboljsi modeli:
    #train_dict_list, train_dict_names, test_dict_list, test_dict_names = logparser.parse_log("/home/niko/caffe-models/diabetic-retinopathy-detection/log/log_normal_pca_small_kernels.txt")
    #train_dict_list, train_dict_names, test_dict_list, test_dict_names = logparser.parse_log("/home/niko/caffe-models/diabetic-retinopathy-detection/cascading/log/stage_0_1234/small_kernels.txt")
    #train_dict_list, train_dict_names, test_dict_list, test_dict_names = logparser.parse_log("/home/niko/caffe-models/diabetic-retinopathy-detection/log/log_small_kernels_size_3.txt")
    #train_dict_list, train_dict_names, test_dict_list, test_dict_names = logparser.parse_log("/home/niko/caffe-models/diabetic-retinopathy-detection/log/log_small_kernels_v3.txt")
    train_dict_list, train_dict_names, test_dict_list, test_dict_names = logparser.parse_log("/home/niko/caffe-models/diabetic-retinopathy-detection/log/deep_v1.txt")
    trainSeconds = [i['Seconds'] for i in train_dict_list]
    trainLearningRate = [i['LearningRate'] for i in train_dict_list]
    trainNumIters = [i['NumIters'] for i in train_dict_list]
    trainingLoss = [i['TrainingLoss'] for i in train_dict_list]
    
    testSeconds = [i['Seconds'] for i in test_dict_list]
    testNumIters = [i['NumIters'] for i in test_dict_list]
    testLoss = [i['TestLoss'] for i in test_dict_list]
    testAccuracy = [i['TestAccuracy'] for i in test_dict_list]
    
    plotLoss(trainNumIters, trainingLoss, testNumIters, testLoss, trainLearningRate, testAccuracy)
    #plotAccuracy(trainNumIters, trainingLoss, testNumIters, testAccuracy)
    
    accuracyMax = 0
    iterMax = 0
    for index, item in enumerate(testNumIters):
        if item % 10000 == 0:
            if testAccuracy[index] > accuracyMax:
                accuracyMax = testAccuracy[index]
                iterMax = testNumIters[index]
    print "Max accuracy: %f, iteration: %d" % (accuracyMax, iterMax)
        
    