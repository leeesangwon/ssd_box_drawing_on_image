# python2.7

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

import glob
from tqdm import tqdm

plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Make sure that caffe is on the python path:
caffe_root = '/home/sangwon/Projects/refinedet/RefineDet'  # this file is expected to be in {caffe_root}/examples
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')

import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

from google.protobuf import text_format
from caffe.proto import caffe_pb2

import math


class Network(object):
    def __init__(self, model_def, model_weights, labelmap_file, image_resize):
        self.labelmap = self.__get_labelmap(labelmap_file)
        self.net = self.__get_net(model_def, model_weights)
        self.transformer = self.__get_transformer()
        self.image_size = image_resize

        # set net to batch size of 1
        self.net.blobs['data'].reshape(1, 3, image_resize, image_resize)

    def __get_labelmap(self, labelmap_file):
        with open(labelmap_file, 'r') as f:
            labelmap = caffe_pb2.LabelMap()
            text_format.Merge(str(f.read()), labelmap)
        return labelmap

    def __get_net(self, model_def, model_weights):
        return caffe.Net(model_def,      # defines the structure of the model
                             model_weights,   # contains the trained weights
                             caffe.TEST)     # use test mode (e.g., don't perform dropout)

    def __get_transformer(self):
        net = self.net
        # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_mean('data', np.array([104,117,123])) # mean pixel
        transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
        transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
        return transformer

    def forward(self, image):
        transformed_image = self.transformer.preprocess('data', image)
        self.net.blobs['data'].data[...] = transformed_image

        # Forward pass.
        detections = self.net.forward()['detection_out']

        return detections

    def forward_center(self, image):
        image_cropper = ImageCropper(image)
        image = image_cropper.get_image_center()
        detections = self.forward(image)
        detections = image_cropper.align_coordinates(detections)

        return detections


class ImageCropper(object):
    def __init__(self, image, grid=(3, 5)):
        self.image = image
        self.grid = grid  # (h, w)
        self.center = (int(x/2) for x in grid)

        self.image_grid = self.__crop_image_by_grid()

    def get_cropped_image(self, grid_idx):
        return self.img_list[grid_idx[0]][grid_idx[1]]

    def get_image_center(self):
        return self.get_cropped_image(self.center)

    def align_coordinates(self, detections):
        y_grid, x_grid = self.grid
        x_offset = 1. / x_grid * 2
        y_offset = 1. / y_grid * 1
        detections[:,:,:,3] = detections[:,:,:,3] / x_grid + x_offset  # xmin
        detections[:,:,:,4] = detections[:,:,:,4] / y_grid + y_offset  # ymin
        detections[:,:,:,5] = detections[:,:,:,5] / x_grid + x_offset  # xmax
        detections[:,:,:,6] = detections[:,:,:,6] / y_grid + y_offset  # ymax
        return detections

    def __crop_image_by_grid(self):
        """Crop image by grid, and return as mxn array of image.
        If size of image was not divisible by grid, images of last row and column
        would be smaller than others.
        Args
            img: numpy array
            grid: a tuple (m, n)
        """
        img = self.image
        grid = self.grid
        w, h = img.shape[0:2]
        w_crop = math.ceil(w / grid[0])
        h_crop = math.ceil(h / grid[1])
        img_list = []
        for c in range(grid[0]):
            column_list = []
            for r in range(grid[1]):
                left = int(c * w_crop)
                right = int(min(left + w_crop, w))
                upper = int(r * h_crop)
                lower = int(min(upper + h_crop, h))
                column_list.append(img[left:right, upper:lower, :])
            img_list.append(column_list)
        return img_list


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


def draw_bbox_on_image(image, detections, output_image_path, labelmap):
    # Parse the outputs.
    det_label = detections[0,0,:,1]
    det_conf = detections[0,0,:,2]
    det_xmin = detections[0,0,:,3]
    det_ymin = detections[0,0,:,4]
    det_xmax = detections[0,0,:,5]
    det_ymax = detections[0,0,:,6]

    # Get detections with confidence higher than 0.6.
    top_conf_indices = [(i, conf) for i, conf in enumerate(det_conf) if conf >= 0.6]

    # Sort using conf
    top_indices = [i for i, conf in sorted(top_conf_indices, key=lambda x: x[1], reverse=True)]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_labels = get_labelname(labelmap, top_label_indices)
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

    # draw bounding boxes
    drawing_labels = ['bicycle', 'bus', 'car', 'motorbike', 'person', 'train']
    h, w = image.shape[0:2]
    dpi = int(h/10)
    figsize = (float(w)/dpi, float(h)/dpi)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(image, aspect='equal')
    ax.axis('off') 

    for i in xrange(top_conf.shape[0]):
        label = int(top_label_indices[i])
        label_name = top_labels[i]
        if label_name not in drawing_labels:
            continue
        xmin = int(round(top_xmin[i] * image.shape[1]))
        ymin = int(round(top_ymin[i] * image.shape[0]))
        xmax = int(round(top_xmax[i] * image.shape[1]))
        ymax = int(round(top_ymax[i] * image.shape[0]))
        score = top_conf[i]
        display_txt = '%s: %.2f'%(label_name, score)
        coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
        color = colors[label]
        ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        ax.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})
        
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.savefig(output_image_path, transparent=True)
    ax.clear()
    plt.close(fig)


def draw_bbox_on_images(input_list, output_folder, labelmap_file, model_def, model_weights, image_resize, use_center=False):
    assert input_list, "input_folder(%s) doesn't have image file(jpeg or png format)" % input_folder
    for input_image in tqdm(input_list):
        network = Network(model_def, model_weights, labelmap_file, image_resize)
        output_image = os.path.join(output_folder, os.path.basename(input_image))
        image = caffe.io.load_image(input_image)  # image set path, numpy array

        detections = network.forward(image)
        if use_center:
            detections_center = network.forward_center(image)
            detections = np.concatenate((detections_center, detections), axis=2)

        draw_bbox_on_image(
            image=image,
            detections=detections,
            output_image_path=output_image,
            labelmap=network.labelmap)


def main(input_folder,      # a folder which has input images
         output_folder,     # a folder to save outputs
         labelmap_file,     # path to labelmap file
         model_def,         # path to deploy.prototxt
         model_weights,     # path to .caffemodel
         image_resize,      # size to resize input images
         use_center):       # if True, detection is conducted again using center region of input image
    assert os.path.isdir(input_folder), "input_folder(%s) does not exist" % input_folder
    assert os.path.isfile(labelmap_file), "labelnap_file(%s) does not exist" % labelmap_file
    assert os.path.isfile(model_def), "model_def(%s) does not exist" % model_def
    assert os.path.isfile(model_weights), "model_weights(%s) does not exist" % model_weights
    
    try:
        os.mkdir(output_folder)
    except:
        pass

    input_list = glob.glob(os.path.join(input_folder, "*.jpg"))
    input_list += glob.glob(os.path.join(input_folder, "*/*/*.jpg"))
    input_list += glob.glob(os.path.join(input_folder, "*.JPG"))
    input_list += glob.glob(os.path.join(input_folder, "*.png"))
    input_list += glob.glob(os.path.join(input_folder, "*.PNG"))

    input_list.sort()

    draw_bbox_on_images(input_list, output_folder, labelmap_file, model_def, model_weights, image_resize, use_center)


if __name__=="__main__":
    project_dir = "/home/sangwon/Projects/draw_detection_result"
    model_folder = project_dir + "/models/refinedet320_coco_voc0712plus"
    input_folder = project_dir + "/target_image"
    output_folder= project_dir + "/result_image/refinedet320_plus_coco"
    labelmap_file = caffe_root + "/data/VOC0712/labelmap_voc.prototxt"
    model_def = model_folder + '/deploy.prototxt'
    model_weights = model_folder + '/final.caffemodel'
    image_resize = 320

    for city in ['pedestrian_30m', 'car_60m']:
        main(input_folder=os.path.join(input_folder, city),
             output_folder=os.path.join(output_folder, city),
             labelmap_file=labelmap_file,
             model_def=model_def,
             model_weights=model_weights,
             image_resize=image_resize,
             use_center=True)
