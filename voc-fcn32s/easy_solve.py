import caffe
import surgery, score

import numpy as np
import os
import sys

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass

weights = '../ilsvrc-nets/vgg16-fcn.caffemodel'
architecture = '../ilsvrc-nets/vgg16-fcn.prototxt'

# init
caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

solver = caffe.SGDSolver('easy_solver.prototxt')

# solver.net.copy_from(weights)  # APD: Was getting shape errors from this.
# Apparently need to copy from VGG original format to FCN format.
base_net = caffe.Net(architecture, weights, caffe.TEST)
surgery.transplant(solver.net, base_net)
del base_net

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
val = np.loadtxt('../data/pascal/easy_VOC2011/seg11valid.txt', dtype=str)

for _ in range(25):
    solver.step(4000)
    score.seg_tests(solver, False, val, layer='score')
