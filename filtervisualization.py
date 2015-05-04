'''
Created on Mar 7, 2015

@author: niko
'''

import numpy as np
import matplotlib.pyplot as plt
from common import *

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'



def initializeModel(mdl, pretrainedMdl, binaryProtoFile, imageDims):
    a = caffe.io.caffe_pb2.BlobProto();
    binaryProtoFile = open(binaryProtoFile, 'rb')
    data = binaryProtoFile.read()
    a.ParseFromString(data)
    means=a.data
    means=np.asarray(means)
    means=means.reshape(3,256,256)#change to 256 in case of 256 version
    
    # Set the right path to your model definition binaryProtoFile, pretrained model weights,
    # and the image you would like to classify.
    caffe.set_phase_test()
    #caffe.set_mode_cpu()
    caffe.set_mode_gpu()
    net = caffe.Classifier(mdl, pretrainedMdl,
                           mean = means,
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=imageDims)
    return net

def vis_square(data, padsize=1, padval=0):
        data -= data.min()
        data /= data.max()
        
        # force the number of filters to be square
        n = int(np.ceil(np.sqrt(data.shape[0])))
        padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
        data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
        
        # tile the filters into an image
        data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
        data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
        
        plt.imshow(data)
        plt.show()
        
        #plt.imshow(net.deprocess('data', net.blobs['data'].data[4]))
        #plt.show()
        
        # the parameters are a list of [weights, biases]

    
if __name__ == '__main__':
    mdl = '/home/niko/caffe-models/diabetic-retinopathy-detection/lenet_small_kernels.prototxt'
    pretrainedMdl = '/home/niko/caffe-models/diabetic-retinopathy-detection/snapshot/run-normal_compare_strides/lenet_pca_small_kernels_iter_210000.caffemodel'
    SELECTED_FOLDER, SOURCE_IMAGES_FOLDER_TRAIN, SOURCE_IMAGES_FOLDER_TEST, DATA_IMAGES_TRAIN, DATA_IMAGES_TEST, DATA_IMAGES_TEST_AUGMENTED, TRAIN_LABELS_FILE, TEST_LABELS_FILE, BINARY_PROTO_FILE = getPathsForConfig('run-normal')
    net = initializeModel(mdl, pretrainedMdl, BINARY_PROTO_FILE, imageDims=(227,227))
    
    for imageFile in IMAGE_FILES:
        scores = net.predict([caffe.io.load_image(imageFile)])
        print [(k, v.data.shape) for k, v in net.blobs.items()]
        print [(k, v[0].data.shape) for k, v in net.params.items()]
        
        # take an array of shape (n, height, width) or (n, height, width, channels)
        #  and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
        
        filters = net.params['conv1'][0].data
        vis_square(filters.transpose(0, 2, 3, 1))
        
        feat = net.blobs['conv1'].data[4, :40]
        vis_square(feat, padval=1)
        
        filters = net.params['conv2'][0].data
        #('conv2', (256, 48, 5, 5)),
        #vis_square(filters[:48].reshape(48**2, 5, 5))
        #vis_square(filters[:16].reshape(16**2, 12, 12))
        #vis_square(filters[:20].reshape(20**2, 22, 22))
        vis_square(filters[:20].reshape(20**2, 4, 4))
        
        #feat = net.blobs['conv2'].data[0, :36]
        feat = net.blobs['conv2'].data[0, :96]
        vis_square(feat, padval=1)
        
        #feat = net.blobs['conv3'].data[4]
        #vis_square(feat, padval=0.5)
        
        feat = net.blobs['conv4'].data[4]
        vis_square(feat, padval=0.5)
        
        #feat = net.blobs['conv5'].data[4]
        #vis_square(feat, padval=0.5)
        
        feat = net.blobs['pool4'].data[4]
        vis_square(feat, padval=1)
        
        continue
        
        feat = net.blobs['fc6'].data[4]
        plt.subplot(2, 1, 1)
        plt.plot(feat.flat)
        plt.subplot(2, 1, 2)
        _ = plt.hist(feat.flat[feat.flat > 0], bins=100)
        
        plt.show()
        
        feat = net.blobs['fc7'].data[4]
        plt.subplot(2, 1, 1)
        plt.plot(feat.flat)
        plt.subplot(2, 1, 2)
        _ = plt.hist(feat.flat[feat.flat > 0], bins=100)
        plt.show()
        
        feat = net.blobs['prob'].data[4]
        plt.plot(feat.flat)
        plt.show()
        
        labels = LABELS
        
        # sort top k predictions from softmax output
        top_k = net.blobs['prob'].data[4].flatten().argsort()[-1:-6:-1]
        print [labels[i] for i in top_k]