'''Format of vKITTI Labels

---
info.txt
---
Header: trackID label model color

trackID: track identification number (unique for each object instance)
label: KITTI-like name of the ‘type’ of the object (Car, Van)
model: the name of the 3D model used to render the object (can be used for fine-grained recognition)
color: the name of the color of the object

---
bbox.txt
---
Header: frame cameraID trackID left right top bottom number_pixels truncation_ratio occupancy_ratio isMoving

frame: frame index in the video (starts from 0)
cameraID: 0 (left) or 1 (right)
trackID: track identification number (unique for each object instance)
left, right, top, bottom: KITTI-like 2D ‘bbox’, bounding box in pixel coordinates
(inclusive, (0,0) origin is on the upper left corner of the image)
number_pixels: number of pixels of this vehicle in the image
truncation_ratio: object 2D truncation ratio in [0..1] (0: no truncation, 1: entirely truncated)
occupancy_ratio: object 2D occupancy ratio (fraction of non-occluded pixels) in [0..1]
(0: fully occluded, 1: fully visible, independent of truncation)
isMoving: 0/1 flag to indicate whether the object is really moving between this frame and the next one
'''

import numpy as np
import glob

from src.dacdc_utils import *

DACDC_LABELS = '../../config/labels.txt'
vKITTI_DATA_PATH = '/home/shubham/workspace/dataset/vKITTI/'
DACDC_LABELS_OUT = vKITTI_DATA_PATH+'dacdc_labels.txt'


# get a list of all the label files
vkitti_bbox_fnames = sorted(glob.glob(vKITTI_DATA_PATH+'*/*/bbox.txt'))

def vkitti_trackID_to_dacdc_label(track_id, trackID_label_map):
    label = trackID_label_map[track_id]
    if label in label_map.keys():
        return label
    else:
        return 'Unknown'

# open the file to write labels
f_dacdc_labels = open(DACDC_LABELS_OUT, 'w')

print('generating labels...')
for idx, bbox_fname in enumerate(vkitti_bbox_fnames):
    info_fname = bbox_fname[:-8]+'info.txt'

    # Open info txt file
    with open(info_fname, 'r') as file:
        info_str = file.readlines()
    
    # track id to label map
    trackID_label_map = {}
    for i in range(1,len(info_str)):
        line = info_str[i].replace('\n', '').split()
        trackID = int(line[0])
        label = str(line[1])
        trackID_label_map[trackID] = label

    fnames = []
    bbox2d_exists = []

    # Open annotated 2D bounding-boxes
    with open(bbox_fname, 'r') as file:
        bbox2d_str = file.readlines()

    # Set up a list in which each element specifies
    # if bounding box for that file index exists
    bbox2d_last_idx = int(bbox2d_str[len(bbox2d_str)-1].replace('\n', '').split()[0])
    bbox2d_list = [[] for i in range(bbox2d_last_idx+1)] 
    bbox2d_exists = [False for i in range(bbox2d_last_idx+1)] 
    fnames = ['' for i in range(bbox2d_last_idx+1)] 

    for i in range(1,len(bbox2d_str)):
        line = bbox2d_str[i].replace('\n', '').split()
        trackID_list = []
        bbox_list = []
        if(int(line[1]) == 0): # camera 0
            bbox2d_exists[int(line[0])] = True
            trackID_list.append(int(line[2]))
            bbox_list.append([float(j) for j in line[3:7]])
        for j, bbox in enumerate(bbox_list):
            label = vkitti_trackID_to_dacdc_label(trackID_list[j], trackID_label_map)
            [xmin, xmax, ymin, ymax] = bbox
            bbox2d_list[int(line[0])].append([label, xmin, ymin, xmax, ymax])
        
        fnames[int(line[0])] = bbox_fname[len(vKITTI_DATA_PATH):-8]+'frames/rgb/Camera_0/rgb_' + str(line[0]).zfill(5) + '.jpg'

    for i in range(0,len(bbox2d_exists)):
        if bbox2d_exists[i] == True:
            f_dacdc_labels.write(fnames[i]+' ')
            f_dacdc_labels.write(str(len(bbox2d_list[i]))+' ')
            for bbox in bbox2d_list[i]:
                f_dacdc_labels.write(bbox[0]+' '+str(bbox[1])+' '+str(int(bbox[2]))+' '+str(int(bbox[3]))+' '+str(int(bbox[4]))+' ')

            f_dacdc_labels.write('\n')

print('labels written to file: {}'.format(DACDC_LABELS_OUT))
f_dacdc_labels.close()