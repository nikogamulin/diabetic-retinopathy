'''
Created on 6. jun. 2015

@author: GamulinN
'''

import common

exampleLine = "run-normal_12_left.jpeg 12"
imageListSource = "/media/niko/data/data/DiabeticRetinopathy/labelsTest.txt"
imageListDestination = "/media/niko/data/data/DiabeticRetinopathy/labelsTest_run-normal.txt"
imagesToInclude = ["run-normal"]
itemsDestination = []
imagesCount, itemsSource = common.getItemsFromFile(filename = imageListSource, excludeHeader = False)
for item in itemsSource:
    for configTag in imagesToInclude:
        if configTag in item:
            common.storeItem(item, imageListDestination)
            
print "Process completed, list stored to %s." % imageListDestination
    

