import configparser
import os
from DACDC_DataLoader import DACDC_DataLoader
from src.dacdc_utils import *
from src.configs import *

# this function computes anchor boxes
def compute_anchor_boxes(annotations):
    # list of parameters
    width, height, center_x, center_y = [], [], [], []

    # iterate through all the bounding-boxes
    for annotation in annotations:
        for bbox in annotation:
            width.append(float(bbox[2] - bbox[0]))
            height.append(float(bbox[3] - bbox[1]))
            center_x.append(float(bbox[0]) + (float(bbox[2] - bbox[0])/2.0))
            center_y.append(float(bbox[1]) + (float(bbox[3] - bbox[1])/2.0))
        
    # perform k-means clustering
    X = np.zeros((len(width), 2), dtype=np.float32)
    X[:,0] = np.asarray(width, dtype=np.float32) / original_img_w
    X[:,1] = np.asarray(height, dtype=np.float32) / original_img_h
    kmeans = KMeans(n_clusters=n_anchors, random_state=0).fit(X)

    # sort clusters based on area
    cluster_areas = []
    for cluster in kmeans.cluster_centers_:
        cluster_areas.append(cluster[0]*cluster[1])
    sorted_cluster_centers = [x for _,x in sorted(zip(cluster_areas,kmeans.cluster_centers_))]
    sorted_cluster_centers = (np.asarray(sorted_cluster_centers)).tolist()

    # write to file
    with open(anchor_list_file, 'w') as f:
        for cluster in sorted_cluster_centers:
            print('{:.4f} {:.4f}'.format(cluster[0], cluster[1]), file=f)

    print('---------------')
    print('Anchor Boxes:')
    print('---------------')
    os.system('cat '+anchor_list_file)

if __name__ == '__main__':
    dacdc_dloader = DACDC_DataLoader( data_path, annotated_bbox2d_file)

    _, _, annotations, _, _, _ = \
        dacdc_dloader.assemble_annotations(n_samples=-1, test_size=0.0)

    # compute anchor boxes
    compute_anchor_boxes(annotations)

    # read the anchor boxes from file
    anchor_boxes = None
    with open(anchor_list_file, 'r') as file:
        anchor_boxes = file.readlines()

    anchors = np.zeros((len(anchor_boxes),2), dtype=np.float32)
    for i in range(len(anchor_boxes)):
        anchor_box_w = float(anchor_boxes[i].replace('\n', '').split()[0])
        anchor_box_h = float(anchor_boxes[i].replace('\n', '').split()[1])
        anchors[i,:] = [anchor_box_w, anchor_box_h]

    # test anchor boxes on one sample
    kmeans = KMeans(n_clusters=n_anchors, random_state=0)
    kmeans.cluster_centers_ = anchors

    w_h = np.asarray([0.3,0.5]).reshape(1,-1)
    anchor_idx = kmeans.predict(w_h)
    print('Predicted Anchor Box Index for {} is: {}'.format(w_h, anchor_idx))