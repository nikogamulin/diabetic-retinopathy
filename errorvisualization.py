'''
Created on Mar 12, 2015

@author: niko
'''

import logparser
from pylab import *
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.merge import merge

def plotLoss(trainNumIters, trainingLoss, testNumIters, testLoss, trainLearningRate, testAccuracy):
    fig, ax1 = plt.subplots()
    pltTrainingError = ax1.plot(trainNumIters, trainingLoss, 'r', label = 'Training')
    pltTestError = ax1.plot(testNumIters, testLoss, 'b', label = 'Test')
    ax1.set_ylabel(r"Loss", color="black")
    ax1.set_xlim([0, max(testNumIters)])
    
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
    
def plotMain(logFile, outputLabels = None):
    if outputLabels is None:
        train_dict_list, train_dict_names, test_dict_list, test_dict_names = logparser.parse_log(logFile)
        dfTraining = pd.DataFrame(train_dict_list, columns=['NumIters', 'LearningRate', 'TrainingLoss'])
        dfTest = pd.DataFrame(test_dict_list, columns=['Seconds', 'NumIters', 'TestLoss', 'TestAccuracy'])
        df = merge(dfTraining, dfTest, how='inner', on='NumIters')
    else:
        df = pd.DataFrame()
        for lbl in outputLabels:
            train_dict_list, train_dict_names, test_dict_list, test_dict_names = logparser.parse_log(logFile, lbl)
            dfTrainingCurrent = pd.DataFrame(train_dict_list, columns=['NumIters', 'LearningRate', 'TrainingLoss'])
            dfTestCurrent = pd.DataFrame(test_dict_list, columns=['Seconds', 'NumIters', 'TestLoss', 'TestAccuracy'])
            mergedCurrent = merge(dfTrainingCurrent, dfTestCurrent, how='inner', on='NumIters')
            if 'NumIters' in df:
                df = merge(df, mergedCurrent, how='inner', on='NumIters')
                df['TrainingLoss'] = df['TrainingLoss_x'] + df['TrainingLoss_y']
                df['TestLoss'] = df['TestLoss_x'] + df['TestLoss_y']
                df = df.drop(['TrainingLoss_x', 'TrainingLoss_y', 'TestLoss_x', 'TestLoss_y'], 1)
            else:
                df = mergedCurrent
                       
    df.plot(x='NumIters', y=['TrainingLoss', 'TestLoss'])
    plt.show()
            
        
        

if __name__ == '__main__':
    f="/home/niko/caffe-models/diabetic-retinopathy-detection/log/log_hdf5_v1.txt"
    plotMain(f)
        
    