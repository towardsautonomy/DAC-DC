'''Format of KITTI Labels

#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
'''

import numpy as np
import glob

from src.dacdc_utils import *

DACDC_LABELS = '../../config/labels.txt'
KITTI_TRAINING_DATA_PATH = '/home/shubham/workspace/dataset/KITTI/data_object_image_2/training/'
DACDC_LABELS_OUT = KITTI_TRAINING_DATA_PATH+'dacdc_labels.txt'
IMG_FOLDER_REL_PATH = 'image_2/'

kitti_labels_fnames_path = KITTI_TRAINING_DATA_PATH+'label_2/'

# get a list of all the label files
kitti_labels_fnames = sorted(glob.glob(kitti_labels_fnames_path+'*.txt'))

def kitti_label_to_dacdc_label(label):
    if label in label_map.keys():
        return label
    else:
        return 'Unknown'

# open the file to write labels
f_dacdc_labels = open(DACDC_LABELS_OUT, 'w')

print('generating labels...')
for idx, label_fname in enumerate(kitti_labels_fnames):
    labels = None
    with open(label_fname, 'r') as file:
        labels = file.readlines()

    file_idx = label_fname[-10:-4]

    f_dacdc_labels.write(IMG_FOLDER_REL_PATH+str(file_idx)+'.png ')
    f_dacdc_labels.write(str(len(labels))+' ')
    for i in range(len(labels)):
        label = kitti_label_to_dacdc_label(str(labels[i].replace('\n', '').split()[0]))
        xmin = int(float(labels[i].replace('\n', '').split()[4]))
        ymin = int(float(labels[i].replace('\n', '').split()[5]))
        xmax = int(float(labels[i].replace('\n', '').split()[6]))
        ymax = int(float(labels[i].replace('\n', '').split()[7]))
        f_dacdc_labels.write(label+' '+str(xmin)+' '+str(ymin)+' '+str(xmax)+' '+str(ymax)+' ')

    f_dacdc_labels.write('\n')

print('labels written to file: {}'.format(DACDC_LABELS_OUT))
f_dacdc_labels.close()