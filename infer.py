import numpy as np
from PIL import Image

import caffe

# Allie added:
import glob
import os
import random
import scipy.misc
import shutil  # for copying
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

if __name__ == '__main__':

    VOC_path = 'data/pascal/VOC2011'
    all_gt_fnames = glob.glob('{}/{}/*.png'.format(VOC_path, 'SegmentationClass'))
    assert len(all_gt_fnames) > 0, 'No files in this path, or path does not exist.'
    random.seed()

    # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
    rand_idx = random.randint(0, len(all_gt_fnames)-1)
    gt_fname = os.path.basename(all_gt_fnames[rand_idx])
    rgb_fname = gt_fname.replace('.png', '.jpg')
    
    im = Image.open('{}/JPEGImages/{}'.format(VOC_path, rgb_fname))
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    in_ = in_.transpose((2,0,1))

    architecture = 'voc-fcn32s/deploy.prototxt'

    for itr in [100000, 8000]:
        weights = 'voc-fcn32s/snapshot/train_iter_{}.caffemodel'.format(itr)

        # load net
        #net = caffe.Net('voc-fcn8s/deploy.prototxt', 'voc-fcn8s/fcn8s-heavy-pascal.caffemodel', caffe.TEST)
        net = caffe.Net(architecture,
                        weights,
                        caffe.TEST)
        #base_net = caffe.Net(architecture, weights, caffe.TEST)
        #surgery.transplant(net, base_net)
        #del base_net

        # shape for input (data blob is N x C x H x W), set data
        net.blobs['data'].reshape(1, *in_.shape)
        net.blobs['data'].data[...] = in_
        # run net and take argmax for prediction
        net.forward()
        out_score = net.blobs['score'].data[0]
        out = net.blobs['score'].data[0].argmax(axis=0)


        # Display
        rgb_out = convert_2d_class_img_to_rgb_img(out)
        rgb_in = np.transpose(net.blobs['data'].data[0], (1, 2, 0))
        save_image(rgb_in, 'input.png')
        save_image(rgb_out, 'prediction_itr_{}.png'.format(itr))
        shutil.copyfile(os.path.join(VOC_path, 'SegmentationClass', gt_fname), 'ground_truth.png')
