import os
import time
import torch
import random
random.seed(777)
import argparse
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (12, 9)  # small images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap
from imageio import imread

conf_thresh = 0.4
MIN_BOXES = 10
MAX_BOXES = 36

predefined_non_filtering_cls = []

with open(
        './data/statistic/non_filtering_classes.txt',
        'r') as f:
    for line in f:
        predefined_non_filtering_cls.append(line[:-1])

bua_data_path = './data/statistic/'
bua_attributes = ['__no_attribute__']
with open(os.path.join(bua_data_path, 'attributes_vocab.txt')) as f:
    for att in f.readlines():
        bua_attributes.append(att.split(',')[0].lower().strip())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vg_dataset', dest='vg_dataset',
                        default='ENV1_train', type=str)
    parser.add_argument('--image_dir', dest='image_dir')
    parser.add_argument('--image_file', dest='image_file')
    parser.add_argument('--image_list_file', dest='image_list_file')
    parser.add_argument('--detection_file', dest='detection_file')
    parser.add_argument('--attr_detection_file', dest='attr_detection_file')
    parser.add_argument('--split_ind', dest='split_ind', default=0, type=int)
    parser.add_argument('--each_image_query', dest='each_image_query', default=10, type=int)
    parser.add_argument('--topn', dest='topn', default=100, type=int)
    parser.add_argument('--out_path', dest='out_path')
    parser.add_argument('--attr_iou_thresh', dest='attr_iou_thresh', default=0.5, type=float)
    parser.add_argument('--attr_conf_thresh', dest='attr_conf_thresh', default=0.4, type=float)
    args = parser.parse_args()
    return args


def filter_detect_cls(cls_name):
    if cls_name in predefined_non_filtering_cls:
        return True
    else:
        return False

def center_of_bbox(bbox):
    return [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2]

def area_of_bbox(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

def overlap_area_of_two_bbox(bbox1, bbox2):
    xmin, xmax = max(bbox1[0], bbox2[0]), min(bbox1[2], bbox2[2])
    ymin, ymax = max(bbox1[1], bbox2[1]), min(bbox1[3], bbox2[3])
    width = xmax - xmin
    height = ymax - ymin
    if width < 0 or height < 0:
        return 0
    else:
        return width * height

def IoU(bbox1, bbox2):
    overlap_area = overlap_area_of_two_bbox(bbox1, bbox2)
    if overlap_area == 0:
        return 0
    else:
        area_bbox1 = area_of_bbox(bbox1)
        area_bbox2 = area_of_bbox(bbox2)
        return overlap_area / (area_bbox1 + area_bbox2 - overlap_area)

def remove_overlap_bbox(descriptor, iou_thresh=0.4, reform2dict=True):
    tmp_descriptor = descriptor.copy()
    pop_ind = []
    for ind1 in range(len(descriptor) - 1):
        for ind2 in range(ind1 + 1, len(descriptor)):
            iou = IoU(descriptor[ind1]['bbox'], descriptor[ind2]['bbox'])
            if iou > iou_thresh:
                if descriptor[ind1]['bbox'][4] < descriptor[ind2]['bbox'][4]:  # keep the one with large confidence
                    if ind1 not in pop_ind:
                        pop_ind.append(ind1)
                else:
                    if ind2 not in pop_ind:
                        pop_ind.append(ind2)

    pop_ind = sorted(pop_ind, reverse=True)
    for ind in pop_ind:
        tmp_descriptor.pop(ind)

    if reform2dict:
        new_descriptor = {}
        for ind in range(len(tmp_descriptor)):
            people_info = tmp_descriptor[ind]
            cls = people_info['class']
            if cls not in new_descriptor:
                new_descriptor[cls] = []
            new_descriptor[cls].append(people_info)

        return new_descriptor
    else:
        return tmp_descriptor


def remove_tiny_bbox(descriptor, image_size, area_thresh=0.001):
    tmp_descriptor = descriptor.copy()
    pop_ind = []
    image_area = image_size[0] * image_size[1]
    for ind in range(len(descriptor)):
        bbox_area = area_of_bbox(descriptor[ind]['bbox'])
        if bbox_area / image_area < area_thresh:
            pop_ind.append(ind)

    pop_ind = sorted(pop_ind, reverse=True)
    for ind in pop_ind:
        tmp_descriptor.pop(ind)

    return tmp_descriptor

def remove_large_bbox(descriptor, image_size, area_thresh=0.07):
    tmp_descriptor = descriptor.copy()
    pop_ind = []
    image_area = image_size[0] * image_size[1]
    for ind in range(len(descriptor)):
        bbox_area = area_of_bbox(descriptor[ind]['bbox'])
        if bbox_area / image_area > area_thresh:
            pop_ind.append(ind)

    pop_ind = sorted(pop_ind, reverse=True)
    for ind in pop_ind:
        tmp_descriptor.pop(ind)

    return tmp_descriptor

def match_attribute_to_object(descriptor, descriptor_with_attr, iou_thresh):
    for key, value in descriptor.items():
        if key in descriptor_with_attr:
            for ind_wo_attr, item_wo_attr in enumerate(value):
                bbox_wo_attr = item_wo_attr['bbox'][:4]
                for item_with_attr in descriptor_with_attr[key]:
                    if item_with_attr['attr'] is not None:
                        bbox_with_attr = item_with_attr['bbox'][:4]
                        iou = IoU(bbox_wo_attr, bbox_with_attr)
                        if iou > iou_thresh:
                            descriptor[key][ind_wo_attr]['attr'] = item_with_attr['attr']

    return descriptor


def relative_spatial_location(descriptor, image_size):

    align_thresh = 30
    
    """
    Relation extraction between objects in different categories
    """
    horizontal_rel_offset = (120, 45)
    vertical_rel_offset = (45, 120)

    # print(descriptor)

    obj_list = []
    for key, value in descriptor.items():
        for item in value:
            if item['attr'] != None: obj = item['attr'] + " " + item['class']
            else: obj = item['class']
            obj_data = item['bbox'].copy()
            obj_data.append((obj_data[0] + obj_data[2])/2) # x-center
            obj_data.append((obj_data[1] + obj_data[3])/2) # y-center
            obj_data.append(obj)
            obj_list.append(obj_data)
            item['relation between cat'] = []

    for i in range(len(obj_list)):
        for j in range(len(obj_list)):
            if i == j: continue
            x_margin = obj_list[i][5] - obj_list[j][5]
            y_margin = obj_list[i][6] - obj_list[j][6]
            
            # check horizontal relationship (next to, on the {left, right} size of)
            if abs(x_margin) < horizontal_rel_offset[0] and abs(y_margin) < horizontal_rel_offset[1] and abs(x_margin) > abs(y_margin):
                obj_list[i].append('next to ' + obj_list[j][7])
                if x_margin < 0:
                    # object[i] is on the left
                    obj_list[i].append('on the left side of ' + obj_list[j][7])
                else:
                    # object[i] is on the right
                    obj_list[i].append('on the right side of ' + obj_list[j][7])

            # check vertical relationship (in front of, behind) 
            if abs(x_margin) < vertical_rel_offset[0] and abs(y_margin) < vertical_rel_offset[1] and abs(y_margin) > abs(x_margin):
                if y_margin < 0:
                    # object[i] is behind of object[j]
                    obj_list[i].append('behind of ' + obj_list[j][7])
                else:
                    # object[i] is in front of object[j]
                    obj_list[i].append('in front of ' + obj_list[j][7])


    obj_list_idx = 0
    for key, value in descriptor.items():
        for item in value:
            assert item['bbox'] == obj_list[obj_list_idx][:5]
            item['relation between cat'] = obj_list[obj_list_idx][8:]        
            obj_list_idx += 1

    """
    Relation extraction between objects in the same categories
    """
    for key, value in descriptor.items(): #descriptor = {'cup': [{'class': cup,'bbox': [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), float(conf)],'spatial': [], 'relation between cat': ["on the right side of yellow box"], attr': detected_attr}, {...}]}
        for ii in range(len(value)): #init 5 relationship information adding into object's dictionary
            descriptor[key][ii]['location'] = []
            descriptor[key][ii]['horizontal relation among cat'] = []
            descriptor[key][ii]['vertical relation among cat'] = []
            descriptor[key][ii]['horizontal relation among cat, att'] = []
            descriptor[key][ii]['vertical relation among cat, att'] = []
            
        ####### absolute location on the table #######
        tmp_bbox_center = []
        for idx in range(len(value)):
            tmp_bbox_center.append(center_of_bbox(value[idx]['bbox']))
        tmp_bbox_center = np.array(tmp_bbox_center)    
        if tmp_bbox_center[idx, 0] > image_size[1] * 2 / 3:
            descriptor[key][idx]['location'].append('right side')
            if tmp_bbox_center[idx, 1] < image_size[0] * 2 / 5:
                descriptor[key][idx]['location'].append('top right')
        elif tmp_bbox_center[idx, 0] < image_size[1] / 3:
            descriptor[key][idx]['location'].append('left side')
            if tmp_bbox_center[idx, 1] < image_size[0] * 2 / 5:
                descriptor[key][idx]['location'].append('top left')
        elif image_size[1] * 2 / 3 > tmp_bbox_center[idx, 0] > image_size[1] / 3:
            descriptor[key][idx]['location'].append('middle')
            descriptor[key][idx]['location'].append('center')
            if tmp_bbox_center[idx, 1] < image_size[0] * 2 / 5:
                descriptor[key][idx]['location'].append('top middle')
                descriptor[key][idx]['location'].append('top center')

        ######## for objects with same category #######
        if len(value) > 1: #more than 1 objects in same class
            tmp_bbox_center = []
            for ind in range(len(value)):
                tmp_bbox_center.append(center_of_bbox(value[ind]['bbox']))
            tmp_bbox_center = np.array(tmp_bbox_center)
            xmin_ind, xmax_ind = np.argmin(tmp_bbox_center[:, 0]), np.argmax(tmp_bbox_center[:, 0])
            ymin_ind, ymax_ind = np.argmin(tmp_bbox_center[:, 1]), np.argmax(tmp_bbox_center[:, 1])
            x_sorted = tmp_bbox_center[:, 0].argsort()
            y_sorted = tmp_bbox_center[:, 1].argsort()
            if (tmp_bbox_center[x_sorted[-1], 0] - tmp_bbox_center[x_sorted[-2], 0]) > align_thresh: 
                descriptor[key][xmax_ind]['horizontal relation among cat'].append('rightmost')            
            if (tmp_bbox_center[x_sorted[1], 0] - tmp_bbox_center[x_sorted[0], 0]) > align_thresh: 
                descriptor[key][xmin_ind]['horizontal relation among cat'].append('leftmost')
            if len(value) == 3:
                for ind in range(len(value)):
                    descriptor[key][xmin_ind]['horizontal relation among cat'].append('left')
                    descriptor[key][xmax_ind]['horizontal relation among cat'].append('right')
                    if ind not in [xmin_ind, xmax_ind]:
                        descriptor[key][ind]['horizontal relation among cat'].append('middle')
                        descriptor[key][ind]['horizontal relation among cat'].append('center')
            elif len(value) == 2:
                descriptor[key][xmin_ind]['horizontal relation among cat'].append('left')
                descriptor[key][xmax_ind]['horizontal relation among cat'].append('right')
            elif len(value) > 3:
                descriptor[key][xmax_ind]['horizontal relation among cat'].append('right')
                descriptor[key][xmin_ind]['horizontal relation among cat'].append('left')
                descriptor[key][x_sorted[int(len(x_sorted)/2)]]['horizontal relation among cat'].append('middle')
                descriptor[key][x_sorted[int(len(x_sorted)/2)]]['horizontal relation among cat'].append('center')
            else:
                pass

            if (tmp_bbox_center[ymax_ind, 1] - tmp_bbox_center[y_sorted[-2], 1]) > align_thresh:
                descriptor[key][ymax_ind]['vertical relation among cat'].append('in front')
            if (-tmp_bbox_center[ymin_ind, 1] + tmp_bbox_center[y_sorted[1], 1]) > align_thresh:
                descriptor[key][ymin_ind]['vertical relation among cat'].append('in behind')

            ########### for objects with same category and attribute ###########
            for kk, sample_of_same_cate in enumerate(value):
                dict_sample = {}
                if sample_of_same_cate['attr'] not in dict_sample.keys(): #make key for first seen attribute and inject whole dictionary
                    dict_sample[sample_of_same_cate['attr']] = [sample_of_same_cate]
                else: #if same attribute exsist!
                    dict_sample[sample_of_same_cate['attr']].append(sample_of_same_cate)
            for a_key, a_val in dict_sample.items():
                if len(a_val) > 1: #more than 1 objects in same class and attribute
                    tmp_bbox_center = []
                    for ind in range(len(a_val)):
                        tmp_bbox_center.append(center_of_bbox(a_val[ind]['bbox']))
                    tmp_bbox_center = np.array(tmp_bbox_center)
                    xmin_ind, xmax_ind = np.argmin(tmp_bbox_center[:, 0]), np.argmax(tmp_bbox_center[:, 0])
                    ymin_ind, ymax_ind = np.argmin(tmp_bbox_center[:, 1]), np.argmax(tmp_bbox_center[:, 1])
                    x_sorted = tmp_bbox_center[:, 0].argsort()
                    y_sorted = tmp_bbox_center[:, 1].argsort()
                    if (x_sorted[0] != xmin_ind) or (y_sorted[0] != ymin_ind):
                        print("!!!!!!!!!!!!wrong sorting")
                        exit()
                    if (tmp_bbox_center[x_sorted[-1], 0] - tmp_bbox_center[x_sorted[-2], 0]) > align_thresh: 
                        descriptor[a_key][xmax_ind]['horizontal relation among cat, att'].append('rightmost')            
                    if (tmp_bbox_center[x_sorted[1], 0] - tmp_bbox_center[x_sorted[0], 0]) > align_thresh: 
                        descriptor[a_key][xmin_ind]['horizontal relation among cat, att'].append('leftmost')
                    if len(a_val) == 3:
                        for ind in range(len(a_val)):
                            descriptor[a_key][xmin_ind]['horizontal relation among cat, att'].append('left')
                            descriptor[a_key][xmax_ind]['horizontal relation among cat, att'].append('right')
                            if ind not in [xmin_ind, xmax_ind]:
                                descriptor[a_key][ind]['horizontal relation among cat, att'].append('middle')
                                descriptor[a_key][ind]['horizontal relation among cat, att'].append('center')
                    elif len(a_val) == 2:
                        descriptor[a_key][xmin_ind]['horizontal relation among cat, att'].append('left')
                        descriptor[a_key][xmax_ind]['horizontal relation among cat, att'].append('right')
                    elif len(a_val) > 3:
                        descriptor[a_key][xmax_ind]['horizontal relation among cat, att'].append('right')
                        descriptor[a_key][xmin_ind]['horizontal relation among cat, att'].append('left')
                        descriptor[a_key][x_sorted[int(len(x_sorted)/2)]]['horizontal relation among cat, att'].append('middle')
                        descriptor[a_key][x_sorted[int(len(x_sorted)/2)]]['horizontal relation among cat, att'].append('center')
                    else:
                        pass

                    if (tmp_bbox_center[ymax_ind, 1] - tmp_bbox_center[y_sorted[-2], 1]) > align_thresh:
                        descriptor[key][ymax_ind]['vertical relation among cat, att'].append('in front')
                    if (-tmp_bbox_center[ymin_ind, 1] + tmp_bbox_center[y_sorted[1], 1]) > align_thresh:
                        descriptor[key][ymin_ind]['vertical relation among cat, att'].append('in behind')                


    return descriptor


def process_of_descriptor(things_descriptor, image_size):
    descriptor = {}
    for key, value in things_descriptor.items():
        descriptor[key] = value
    descriptor = relative_spatial_location(descriptor, image_size)
    return descriptor


def generate_description(descriptor, image_file, pseudo_train_samples, each_image_query):
    all_candidate = []
    for object in descriptor: #object = {'class': , ....} -> dict
        location_candidate = []
        hor_rel_among_cat_candidate = []
        ver_rel_among_cat_candidate = []
        hor_rel_among_cat_att_candidate = []
        ver_rel_among_cat_att_candidate = []
        rel_between_cat_candidate = []

        for ind in range(len(object['location'])):
            location_candidate.append(object['location'][ind])
        for ind in range(len(object['horizontal relation among cat'])):
            hor_rel_among_cat_candidate.append(object['horizontal relation among cat'][ind])
        for ind in range(len(object['vertical relation among cat'])):
            ver_rel_among_cat_candidate.append(object['vertical relation among cat'][ind])
        for ind in range(len(object['horizontal relation among cat, att'])):
            hor_rel_among_cat_att_candidate.append(object['horizontal relation among cat, att'][ind])
        for ind in range(len(object['vertical relation among cat, att'])):
            ver_rel_among_cat_att_candidate.append(object['vertical relation among cat, att'][ind])
        for ind in range(len(object['relation between cat'])):
            rel_between_cat_candidate.append(object['relation between cat'][ind])
            #rel_between_cat_candidate.append(object['vertical relation among cat, att'][ind]) # error ocurring (index out of range)

        ### Template 1: attr + noun
        if object['attr'] is not None:
            description_string = '{} {}'.format(object['attr'], object['class'])
        #else:
        #    description_string = '{}'.format(object['class'])
            tmp_pseudo_train_sample = [image_file, 'useless placeholder', object['bbox'][:4],
                                    description_string, 'useless placeholder']
            all_candidate.append(tmp_pseudo_train_sample)

        ### Template 2,3: (noun location) and (attr noun location)
        for ind in range(len(location_candidate)):
            if object['attr'] is not None:
                description_string = '{} {} on the {} of the table'.format(object['attr'], object['class'], location_candidate[ind])
            else:
                description_string = '{} on the {} of the table'.format(object['class'], location_candidate[ind])
            tmp_pseudo_train_sample = [image_file, 'useless placeholder', object['bbox'][:4],
                                       description_string, 'useless placeholder']
            all_candidate.append(tmp_pseudo_train_sample)

        ### Template 4,5
        for ind in range(len(hor_rel_among_cat_candidate)):
            if object['attr'] is not None:
                description_string = '{} {} {}'.format(hor_rel_among_cat_candidate[ind], object['attr'], object['class'])
            else:
                description_string = '{} {}'.format(hor_rel_among_cat_candidate[ind], object['class'])
            tmp_pseudo_train_sample = [image_file, 'useless placeholder', object['bbox'][:4],
                                       description_string, 'useless placeholder']
            all_candidate.append(tmp_pseudo_train_sample)

        ### Template 6,7
        for ind in range(len(ver_rel_among_cat_candidate)):
            if object['attr'] is not None:
                description_string = '{} {} {}'.format(object['attr'], object['class'], ver_rel_among_cat_candidate[ind])
            else:
                description_string = '{} {}'.format(object['class'], ver_rel_among_cat_candidate[ind])
            tmp_pseudo_train_sample = [image_file, 'useless placeholder', object['bbox'][:4],
                                       description_string, 'useless placeholder']
            all_candidate.append(tmp_pseudo_train_sample)

        ### Template 8
        for ind in range(len(hor_rel_among_cat_att_candidate)):
            if object['attr'] is not None:
                description_string = '{} {} {}'.format(hor_rel_among_cat_att_candidate[ind], object['attr'], object['class'])
                tmp_pseudo_train_sample = [image_file, 'useless placeholder', object['bbox'][:4],
                                        description_string, 'useless placeholder']
                all_candidate.append(tmp_pseudo_train_sample)

        ### Template 9
        for ind in range(len(ver_rel_among_cat_att_candidate)):
            if object['attr'] is not None:
                description_string = '{} {} {}'.format(object['attr'], object['class'], ver_rel_among_cat_att_candidate[ind])
                tmp_pseudo_train_sample = [image_file, 'useless placeholder', object['bbox'][:4],
                                        description_string, 'useless placeholder']
                all_candidate.append(tmp_pseudo_train_sample)

        ### Template 10,11
        for ind in range(len(rel_between_cat_candidate)):
            if object['attr'] is not None:
                description_string = '{} {} {}'.format(object['attr'], object['class'], rel_between_cat_candidate[ind])
            else:
                description_string = '{} {}'.format(object['class'], rel_between_cat_candidate[ind])
            tmp_pseudo_train_sample = [image_file, 'useless placeholder', object['bbox'][:4],
                                       description_string, 'useless placeholder']
            all_candidate.append(tmp_pseudo_train_sample)

    instruction1 = []
    instruction2 = []
    instruction3 = []

    #### add instruction pick, grasp, fetch
    for sample in all_candidate:
        string = sample[3]
        i1 = sample.copy()
        i2 = sample.copy()
        i3 = sample.copy()
        i1[3] = 'pick the ' + string
        i2[3] = 'grasp the ' + string
        i3[3] = 'fetch the ' + string
        instruction1.append(i1)
        instruction2.append(i2)
        instruction3.append(i3)
    all_candidate = all_candidate + instruction1 + instruction2 + instruction3

    if len(all_candidate) < each_image_query:
        print("Number of query generated .... {} !!!!!!!".format(len(all_candidate)))
        return descriptor, pseudo_train_samples + all_candidate
    else:
        tmp_candidate = random.sample(all_candidate, each_image_query)
        return descriptor, pseudo_train_samples + tmp_candidate


def topn_conf_samples(descriptor, topn = 100):

    topn_thing_samples = []
    samples_conf = []
    for key, value in descriptor.items():
        for ind in range(len(value)):
            topn_thing_samples.append(value[ind])
            samples_conf.append(value[ind]['bbox'][-1])

    if len(topn_thing_samples) > 100: # topn -> 100
        delte_sample_ind = np.argsort(np.array(samples_conf))[:len(topn_thing_samples) - topn]
        sorted_delte_sample_ind = sorted(delte_sample_ind, reverse=True)

        for ind in sorted_delte_sample_ind:
            topn_thing_samples.pop(ind)

    return topn_thing_samples #[{obj1}, {obj2}, ...]


def bua_attr_detect(image_attr_detection_result, attr_conf_thresh):
    things_descriptor = []

    for object in image_attr_detection_result:
        cls = object[0]
        attr_conf = object[-1]
        if filter_detect_cls(cls):
            bbox = object[1][:4]
            conf = object[1][-1]
            if bbox[0] == 0:
                bbox[0] = 1
            if bbox[1] == 0:
                bbox[1] = 1

            if np.max(attr_conf) > attr_conf_thresh:
                attr_ind = np.argmax(attr_conf)
                detected_attr = bua_attributes[attr_ind + 1]
                if detected_attr in cls: #if cls is yellow cup, change to cup
                    cls = cls.replace('{} '.format(detected_attr), '')
            else:
                detected_attr = None

            if bbox[0] > bbox[2] or bbox[1] > bbox[3]:
                print('### Warning ###: Unvalid bounding box = {}, class = {}, conf = {}'.format(bbox, cls, conf))
            else:
                thing_info = {'class': cls,
                                'bbox': [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), float(conf)],
                                'spatial': [], 'attr': detected_attr}
                things_descriptor.append(thing_info)
        # else:
        #     print('Ignore the class {}!'.format(cls))

    return things_descriptor


def object_detect(image_object_detection_result):
    things_descriptor = []

    for object in image_object_detection_result:
        cls = object[0]
        if filter_detect_cls(cls):
            
            bbox = object[1][:4]
            conf = object[1][-1]
            if bbox[0] == 0:
                bbox[0] = 1
            if bbox[1] == 0:
                bbox[1] = 1

            if bbox[0] > bbox[2] or bbox[1] > bbox[3]:
                print('### Warning ###: Unvalid bounding box = {}, class = {}, conf = {}'.format(bbox, cls, conf))
            else:
                thing_info = {'class': cls,
                                'bbox': [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), float(conf)],
                                'attr': None, 'spatial': []}
                things_descriptor.append(thing_info)
                
        # else:
        #     print('Ignore the class {}!'.format(cls))

    return things_descriptor


if __name__ == '__main__':
    args = parse_args()

    if args.vg_dataset == 'ENV1_train':
        args.image_dir = '../data/train/ENV1_train'
    elif args.vg_dataset == 'ENV2_train':
        args.image_dir = '../data/train/ENV2_train'    
    print(args.image_dir)

    args.out_path = './data/pseudo_samples/{}'.format(args.vg_dataset)

    args.image_list_file = './data/statistic/{}/{}_train_imagelist_split{}.txt'.format(
        args.vg_dataset, args.vg_dataset, args.split_ind)
    args.detection_file = './data/detection_results/{}/r101_object_detection_results/{}_train_pseudo_split{}_detection_results.pth'.format(
        args.vg_dataset, args.vg_dataset, args.split_ind)
    args.attr_detection_file = './data/detection_results/{}/r152_attr_detection_results/{}_train_pseudo_split{}_attr_detection_results.pth'.format(
        args.vg_dataset, args.vg_dataset, args.split_ind)

    train_image_list = open(args.image_list_file, 'r')
    train_image_files = train_image_list.readlines()
    off_the_shelf_object_detection_result = torch.load(args.detection_file)
    off_the_shelf_attr_detection_result = torch.load(args.attr_detection_file)
    pseudo_train_samples = []
    count = 0
    start_time = time.time()
    for image_ind, image_file in enumerate(train_image_files):
        if image_ind % 100 == 0:
            left_time = ((time.time() - start_time) * (len(train_image_files) - image_ind - 1) / (image_ind + 1)) / 3600
            print('Processing {}-th image, Left Time = {:.2f} hour ...'.format(image_ind, left_time))

        if "\n" in image_file:
            args.image_file = image_file[:-1]
        else:
            args.image_file = image_file
        im_file = os.path.join(args.image_dir, args.image_file)
        print(image_file)
        print(args.image_file)
        print(im_file)

        im = np.array(imread(im_file))
        image_object_detection_result = off_the_shelf_object_detection_result[args.image_file]
        image_attr_detection_result = off_the_shelf_attr_detection_result[args.image_file]

        things_descriptor = object_detect(image_object_detection_result)
        things_descriptor = remove_large_bbox(things_descriptor, im.shape[:2])
        things_descriptor = remove_overlap_bbox(things_descriptor)

        bua_things_descriptor = bua_attr_detect(
            image_attr_detection_result, attr_conf_thresh=args.attr_conf_thresh)
        bua_things_descriptor = remove_overlap_bbox(bua_things_descriptor)

        things_descriptor = match_attribute_to_object(things_descriptor, bua_things_descriptor,
                                                      iou_thresh=args.attr_iou_thresh)

        descriptor = process_of_descriptor(things_descriptor, im.shape[:2])  # image size: (h, w)
        descriptor = topn_conf_samples(descriptor, topn=args.topn)
        descriptor, pseudo_train_samples = generate_description(descriptor, args.image_file, pseudo_train_samples,
                                                                each_image_query=args.each_image_query)
    image_list_file = args.image_list_file

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    torch.save(pseudo_train_samples,
               os.path.join(args.out_path, '{}.pth'.format(args.vg_dataset)))
    print('Save file to {}'.format(
        os.path.join(args.out_path, '{}.pth'.format(args.vg_dataset))))
