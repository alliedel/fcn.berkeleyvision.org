import numpy as np
from PIL import Image

import caffe

# Allie added:
import scipy.misc
import sys
sys.path.append('./data/pascal') # for the colormap
import VOClabelcolormap

VOC_COLORMAP = VOClabelcolormap.color_map()

def save_image(image_arr, filename):
    scipy.misc.toimage(image_arr, cmin=0.0, cmax=255).save(filename)

def convert_2d_class_img_to_rgb_img(class_img):
    print(class_img.shape)
    rgb_img = np.empty((class_img.shape[0], class_img.shape[1], 3))
    for r in range(class_img.shape[0]):
        for c in range(class_img.shape[1]):
            try:
                rgb_img[r, c, :] = VOC_COLORMAP[class_img[r, c]]
            except:
                print('problem here:')
                print(class_img[r,c])
                print(VOClabelcolormap.color_map(class_img[r, c]).ravel())
                raise
    return rgb_img


# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
#im = Image.open('pascal/VOC2010/JPEGImages/2007_000129.jpg')
im = Image.open('pascal/VOC2007/JPEGImages/000166.jpg')
in_ = np.array(im, dtype=np.float32)
in_ = in_[:,:,::-1]
in_ -= np.array((104.00698793,116.66876762,122.67891434))
in_ = in_.transpose((2,0,1))

# load net
net = caffe.Net('voc-fcn8s/deploy.prototxt', 'voc-fcn8s/fcn8s-heavy-pascal.caffemodel', caffe.TEST)
# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_
# run net and take argmax for prediction
net.forward()
out_score = net.blobs['score'].data[0]
out = net.blobs['score'].data[0].argmax(axis=0)

rgb_out = convert_2d_class_img_to_rgb_img(out)
save_image(rgb_out, 'filename.png')
