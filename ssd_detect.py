import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')

import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

from google.protobuf import text_format
from caffe.proto import caffe_pb2

# load PASCAL VOC labels
labelmap_file = 'data/VOC0712/labelmap_voc.prototxt'
file = open(labelmap_file, 'r')
labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), labelmap)

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

#model_def = 'models/VGGNet/VOC0712/PPSSD_512x512_ar6_coco/deploy.prototxt'
#model_weights = 'models/VGGNet/VOC0712/PPSSD_512x512_ar6_coco/VGG_VOC2012_PPSSD_512x512_ft_iter_160000.caffemodel'

model_def = 'models/VGGNet/VOC0712/city/deploy.prototxt'
model_weights = 'models/VGGNet/city.caffemodel'


net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104,117,123])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

##############################################################################
# READ & WRITE
##############################################################################
# set net to batch size of 1
#image_resize = 300
image_resize = 512
net.blobs['data'].reshape(1,3,image_resize,image_resize)

#inputlist = open("data/VOC2012/list.txt", 'r') # test image list path
inputlist = open("examples/cityscapes/list/munster.txt", 'r') # test image list path
lines = inputlist.readlines()

##############################################################################
for x in range(len(lines)):
    preline = lines[x]
    test_line = preline.split(" ")[0]
    outline = test_line.split("/")[-1]

    #outline = test_line[-11:]
    #outline = test_line[-16:] # VOC2012
    #outline = test_line[-40:] # city

    image = caffe.io.load_image("/home/younghyunlab/caffe/examples/cityscapes/" + test_line) # image set path
    plt.imshow(image)
    #plt.show() #plot image on screen
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image

    # Forward pass.
    detections = net.forward()['detection_out']
    # Parse the outputs.
    det_label = detections[0,0,:,1]
    det_conf = detections[0,0,:,2]
    det_xmin = detections[0,0,:,3]
    det_ymin = detections[0,0,:,4]
    det_xmax = detections[0,0,:,5]
    det_ymax = detections[0,0,:,6]

    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_labels = get_labelname(labelmap, top_label_indices)
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

    #plt.imshow(image)
    plt.axis('off')

    # get figure and axis handle 
    fig = plt.gcf()
    currentAxis = plt.gca()
    # get axis extent
    extent = currentAxis.get_window_extent().transformed(fig.dpi_scale_trans.inverted())    

    for i in xrange(top_conf.shape[0]):
        xmin = int(round(top_xmin[i] * image.shape[1]))
        ymin = int(round(top_ymin[i] * image.shape[0]))
        xmax = int(round(top_xmax[i] * image.shape[1]))
        ymax = int(round(top_ymax[i] * image.shape[0]))
        score = top_conf[i]
        label = int(top_label_indices[i])
        label_name = top_labels[i]
        display_txt = '%s: %.2f'%(label_name, score)
        coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
        color = colors[label]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})

    #plt.savefig('examples/VOC2012_results/PFPNET_coco' + outline, transparent=True, bbox_inches=extent)
    plt.savefig('examples/cityscapes_results/munster/' + outline, transparent=True, bbox_inches=extent)
    currentAxis.clear()
    

