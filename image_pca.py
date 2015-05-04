'''
Created on 27. mar. 2015

@author: niko
'''
from sklearn import decomposition
from pylab import *
from skimage import data, io, color
import copy

from PIL import Image
from numpy import *
import random


def distortPrincipalComponents(components, randomValues, sigma=0.1):
    componentsCount, dimensionsCount = components.shape
    for i in range(len(randomValues)):
        for j in range(componentsCount):
            components[j][i] *= randomValues[i]
    return components

def distortImage(img, componentsToDistortCount, sigma=0.1):
    result = copy(img)
    w,h,c = img.shape 
    #fig, axes = subplots(nrows = c, ncols = 1) 
    #gray()
    n_comp = 200
    randomValues = [random.gauss(0,sigma) for i in range(componentsToDistortCount)]
    channels = [result[:,:,i] for i in range(c)]
    for i in range(c):
        pca = decomposition.PCA(n_components = n_comp)
        pca.fit(channels[i])
        channel_pca = pca.fit_transform(channels[i])
        distorted = distortPrincipalComponents(channel_pca, randomValues)
        result[:,:,i] = pca.inverse_transform(distorted)
        
        #axes[i].imshow(result[:,:,i])
        #axes[i].set_title('ch')
    #plt.show()
    return result


if __name__ == "__main__":
    link = "/home/niko/datasets/DiabeticRetinopathyDetection/processed/run-normal/no_pca/train_test/31_right.jpeg"
    
    retina = io.imread(link)
    
    retinaDistorted = distortImage(retina, 3)
    
    fig, axes = subplots(nrows = 2, ncols = 1)
    axes[0].imshow(retina)
    axes[0].set_title('original image')
    axes[1].imshow(retinaDistorted)
    axes[1].set_title('pca restored')
    plt.show()


