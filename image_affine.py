'''
Created on Apr 14, 2015

@author: niko
'''

from skimage import io
from skimage import transform as tf
import random
import math
'''
def getShearedImage(img, randomValue=None):
    # Create Afine transform
    if randomValue is None:
        randomValue = random.gauss(0,0.2)
    afine_tf = tf.AffineTransform(shear=randomValue)
    
    # Apply transform to image data
    modified = tf.warp(img, afine_tf)
    return modified
'''

def getShearedImage(img, randomValue=None):
    # Create Afine transform
    degreesRange = 20
    radiansRange = 2*math.pi/360*degreesRange
    if randomValue is None:
        randomValue= random.uniform(-radiansRange,radiansRange)
    afine_tf = tf.AffineTransform(shear=randomValue)
    
    # Apply transform to image data
    modified = tf.warp(img, afine_tf)
    return modified

if __name__ == "__main__":
    # Load the image as a matrix
    image = io.imread("/home/niko/datasets/DiabeticRetinopathyDetection/processed/run-normal/train/19_left.jpeg")
    modified = getShearedImage(image)
    
    # Display the result
    io.imshow(modified)
    io.show()