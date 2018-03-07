import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import glob

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Make sure that caffe is on the python path:
caffe_root = '/home/sangwon/Projects/caffe'  # this file is expected to be in {caffe_root}/examples
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')

import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

from google.protobuf import text_format
from caffe.proto import caffe_pb2


def main(input_folder, output_folder, labelmap_file, model_def, model_weights, image_resize):
    assert os.path.isdir(input_folder), "input_folder(%s) does not exist" % input_folder
    assert os.path.isfile(labelmap_file), "labelnap_file(%s) does not exist" % labelmap_file
    assert os.path.isfile(model_def), "model_def(%s) does not exist" % model_def
    assert os.path.isfile(model_weights), "model_weights(%s) does not exist" % model_weights
    
    try:
        os.mkdir(output_folder)
    except:
        pass
    draw_bbox_on_images(input_folder, output_folder, labelmap_file, model_def, model_weights, image_resize)

    
def draw_bbox_on_images(input_folder, output_folder, labelmap_file, model_def, model_weights, image_resize):
    labelmap = get_labelmap(labelmap_file)
    net = get_net(model_def, model_weights)
    transformer = get_transformer(net)

    # set net to batch size of 1
    net.blobs['data'].reshape(1,3,image_resize,image_resize)

    input_list = glob.glob(os.path.join(input_folder, "*.jpg"))
    input_list += glob.glob(os.path.join(input_folder, "*.png"))
    
    assert input_list, "input_folder(%s) doesn't have image file(jpeg or png format)" % input_folder

    for input_image in input_list:
        output_image = os.path.join(output_folder, os.path.basename(input_image))
        draw_bbox_on_image(
            input_image_path=input_image,
            output_image_path=output_image,
            labelmap=labelmap, net=net, transformer=transformer)


def get_labelmap(labelmap_file):
    with open(labelmap_file, 'r') as f:
        labelmap = caffe_pb2.LabelMap()
        text_format.Merge(str(f.read()), labelmap)
    return labelmap


def get_net(model_def, model_weights):
    return caffe.Net(model_def,      # defines the structure of the model
                     model_weights,   # contains the trained weights
                     caffe.TEST)     # use test mode (e.g., don't perform dropout)


def get_transformer(net):
    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104,117,123])) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
    return transformer


def draw_bbox_on_image(input_image_path, output_image_path, labelmap, net, transformer):
    image = caffe.io.load_image(input_image_path) # image set path
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal')
    ax.imshow(image, aspect='equal')
    ax.axis('off')
    
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

    # get axis extent
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())    

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
        ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        ax.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})
    
    plt.savefig(output_image_path, transparent=True, bbox_inches=extent)
    ax.clear()
    

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

if __name__=="__main__":
    # for image_resize in [300, 512]:
    #     input_folder ="/home/sangwon/Projects/ssd_512_test/target_image"
    #     output_folder="/home/sangwon/Projects/ssd_512_test/result_image_%s_Plus" % image_resize
    #     labelmap_file = '/home/sangwon/Projects/caffe/data/VOC0712/labelmap_voc.prototxt'
    #     model_def = '/home/sangwon/Projects/caffe/models/VGGNet/VOC0712Plus/SSD_%sx%s_ft/deploy.prototxt' % (image_resize, image_resize)
    #     model_weights = '/home/sangwon/Projects/caffe/models/VGGNet/VOC0712Plus/SSD_%sx%s_ft/VGG_VOC0712Plus_SSD_%sx%s_ft_iter_160000.caffemodel' % (image_resize, image_resize, image_resize, image_resize)
    #     image_resize = image_resize
        
    #     for city in ['frankfurt', 'lindau', 'munster']:
    #         main(input_folder=os.path.join(input_folder, city),
    #             output_folder=os.path.join(output_folder, city),
    #             labelmap_file=labelmap_file,
    #             model_def=model_def,
    #             model_weights=model_weights,
    #             image_resize=image_resize)

    input_folder ="/home/sangwon/Projects/ssd_512_test/target_image"
    output_folder="/home/sangwon/Projects/ssd_512_test/result_image_300"
    labelmap_file = '/home/sangwon/Projects/caffe/data/VOC0712/labelmap_voc.prototxt'
    model_def = '/home/sangwon/Projects/caffe/models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt'
    model_weights = '/home/sangwon/Projects/caffe/models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel'
    image_resize = 300
    
    # main(input_folder=input_folder,
    #     output_folder=output_folder,
    #     labelmap_file=labelmap_file,
    #     model_def=model_def,
    #     model_weights=model_weights,
    #     image_resize=image_resize)

    for city in ['frankfurt', 'lindau', 'munster']:
        main(input_folder=os.path.join(input_folder, city),
            output_folder=os.path.join(output_folder, city),
            labelmap_file=labelmap_file,
            model_def=model_def,
            model_weights=model_weights,
            image_resize=image_resize)