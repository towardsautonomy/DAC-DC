import configparser
from dataLoader import *
from configs import *

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

    print('---------------')
    print('Anchor Boxes:')
    print('---------------')
    print(kmeans.cluster_centers_)

    # write to file
    with open(anchor_list_file, 'w') as f:
        for cluster in kmeans.cluster_centers_:
            print('{:.4f} {:.4f}'.format(cluster[0], cluster[1]), file=f)

if __name__ == '__main__':
    dataset='vkitti'
    if dataset == 'vkitti':
        _, annotations, _, _, _, _ = \
            assemble_data_vkitti(n_samples=-1, test_size=0.0, data_path=data_path)
    else:
        _, annotations, _, _, _, _ = \
            assemble_data_bbox_vector(n_samples=-1, test_size=0.0, data_path=data_path)

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