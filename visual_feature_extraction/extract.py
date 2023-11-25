import os
import matplotlib.pyplot as plt

# set display defaults
plt.rcParams['figure.figsize'] = (12, 9)  # small images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

data_path = './evaluation'

# Load classes
classes = ['__background__']
with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
    for object in f.readlines():
        classes.append(object.split(',')[0].lower().strip())

# Load attributes
attributes = ['__no_attribute__']
with open(os.path.join(data_path, 'attributes_vocab.txt')) as f:
    for att in f.readlines():
        attributes.append(att.split(',')[0].lower().strip())

import os
import cv2
import time
import torch
import random
random.seed(777)
import warnings
import argparse
import numpy as np
import detectron2.utils.comm as comm

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.engine import DefaultTrainer, default_setup, launch
from detectron2.data import build_detection_test_loader, build_detection_train_loader

from FasterRCNN.lib.model.roi_layers import nms
from bua import add_bottom_up_attention_config
from utils.extract_utils import get_image_blob


config_file = './configs/caffe/extract-bua-caffe-r152.yaml'

cfg = get_cfg()
add_bottom_up_attention_config(cfg, True)
cfg.merge_from_file(config_file)
cfg.freeze()

MIN_BOXES = 10
MAX_BOXES = 36
CONF_THRESH = 0.4 #used to be 0.4

model = DefaultTrainer.build_model(cfg)
DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
    './output_caffe152/' + cfg.MODEL.WEIGHTS, resume=True
)
model.eval()


def parse_args():
    """
        Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='vg', type=str)
    parser.add_argument('--vg_dataset', dest='vg_dataset',
                        default='ENV1_train', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res152', type=str)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models',
                        default="./output_caffe152/")
    parser.add_argument('--image_dir', dest='image_dir',
                        help='directory to load images for demo',
                        default="../data/train/ENV1_train/")
    parser.add_argument('--image_out_dir', dest='image_out_dir',
                        help='directory to load images for demo',
                        default=None)
    parser.add_argument('--image_file', dest='image_file',
                        help='the file name of load images for demo',
                        default="img1.jpg")
    parser.add_argument('--image_list_file', dest='image_list_file',
                        default="./ENV1_train_train_imagelist_split0.txt")
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--split_ind', dest='split_ind',
                        default=0, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    parser.add_argument('--out_path', dest='out_path',
                        default="../instruction_generation/data/detection_results/ENV1/r152_attr_detection_results")

    args = parser.parse_args()
    return args

predefined_non_filtering_cls = []

with open('../instruction_generation/data/statistic/non_filtering_classes.txt','r') as f:
    for line in f:
        predefined_non_filtering_cls.append(line[:-1])

def filter_detect_cls(cls_name):
    if cls_name in predefined_non_filtering_cls:
        return True
    else:
        return False


if __name__ == '__main__':
    args = parse_args()

    detected_samples = {}
    train_image_list = open(args.image_list_file, 'r')
    train_image_files = train_image_list.readlines()
    pseudo_train_samples = []
    count = 0
    start_time = time.time()
    for image_ind, image_file in enumerate(train_image_files):
        left_time = ((time.time() - start_time) * (len(train_image_files) - image_ind - 1) / (image_ind + 1)) / 3600
        if image_ind % 100 == 0:
            print('Processing {}-th image, Left Time = {:.2f} hour ...'.format(image_ind, left_time))
        if "\n" in image_file:
            args.image_file = image_file[:-1]
        else:
            args.image_file = image_file
        print("processing image file {}".format(args.image_file))
        im = cv2.imread(os.path.join(args.image_dir, args.image_file))
        try:
            dataset_dict = get_image_blob(im, cfg.MODEL.PIXEL_MEAN)

            with torch.set_grad_enabled(False):
                boxes, scores, features_pooled, attr_scores = model([dataset_dict])

            dets = boxes[0].tensor.cpu() / dataset_dict['im_scale']
            scores = scores[0].cpu()
            feats = features_pooled[0].cpu()
            attr_scores = attr_scores[0].cpu()

            max_conf = torch.zeros((scores.shape[0])).to(scores.device)
            for cls_ind in range(1, scores.shape[1]):
                cls_scores = scores[:, cls_ind]
                keep = nms(dets, cls_scores, 0.3)
                max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep],
                                            cls_scores[keep],
                                            max_conf[keep])

            keep_boxes = torch.nonzero(max_conf >= CONF_THRESH).flatten()
            if len(keep_boxes) < MIN_BOXES:
                keep_boxes = torch.argsort(max_conf, descending=True)[:MIN_BOXES]
            elif len(keep_boxes) > MAX_BOXES:
                keep_boxes = torch.argsort(max_conf, descending=True)[:MAX_BOXES]

            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            boxes = dets[keep_boxes].numpy()
            objects = np.argmax(scores[keep_boxes].numpy()[:, 1:], axis=1)
            attr_thresh = 0.1
            attr = np.argmax(attr_scores[keep_boxes].numpy()[:, 1:], axis=1)
            attr_conf = np.max(attr_scores[keep_boxes].numpy()[:, 1:], axis=1)
            all_attr_conf = attr_scores[keep_boxes].numpy()[:, 1:]

            detected_samples[args.image_file] = []
            for i in range(len(keep_boxes)):
                kind = objects[i] + 1
                cls = classes[objects[i] + 1]
                bbox = boxes[i]
                conf = scores[keep_boxes[i], kind].cpu()
                if bbox[0] == 0:
                    bbox[0] = 1
                if bbox[1] == 0:
                    bbox[1] = 1

            for i in range(len(keep_boxes)):
                cls = classes[objects[i] + 1]
                #if filter_detect_cls(cls):
                kind = objects[i] + 1
                bbox = boxes[i]
                conf = scores[keep_boxes[i], kind].cpu()
                if bbox[0] == 0:
                    bbox[0] = 1
                if bbox[1] == 0:
                    bbox[1] = 1

                if bbox[0] > bbox[2] or bbox[1] > bbox[3]:
                    print('### Warning ###: Unvalid bounding box = {}, class = {}, conf = {}'.format(bbox, cls, conf))
                else:
                    tmp_sample = [cls, [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), float(conf)], all_attr_conf[i]]
                    detected_samples[args.image_file].append(tmp_sample)
        except:
            print("file error!!!!")
    image_list_file = args.image_list_file
    output_path = args.out_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    torch.save(detected_samples, os.path.join(output_path, '{}_train_pseudo_split{}_attr_detection_results.pth'.format(args.vg_dataset,args.split_ind)))
    print('Save file to {}'.format(os.path.join(output_path, '{}_train_pseudo_split{}_attr_detection_results.pth'.format(args.vg_dataset, args.split_ind))))