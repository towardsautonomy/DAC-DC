import numpy as np
from sklearn.cluster import KMeans
import math
from src.configs import *

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

def get_anchor_idx(w_h):
    anchor_found = [False]*n_anchors
    anchor_idx = [-1]*len(w_h)
    for i in range(len(w_h)):
        mismatch_scores = []
        for j in range(len(anchors)):
            mismatch_score = ((anchors[j][0]-w_h[i][0]) ** 2) + ((anchors[j][1]-w_h[i][1]) ** 2)
            mismatch_scores.append(mismatch_score)

        while(True):
            argmin_idx = np.argmin(np.asarray(mismatch_scores))
            # print(argmin_idx, mismatch_scores)
            if(anchor_found[argmin_idx] == False):
                anchor_found[argmin_idx] = True
                anchor_idx[i] = argmin_idx
                break
            elif(np.count_nonzero(anchor_found) != len(anchor_found)):
                mismatch_scores[argmin_idx] = 1e5
            else:
                break

    return anchor_idx

# assemble label from bounding-boxes
def assemble_label(bboxes, img_size):
    bboxes_copy = bboxes.copy()
    cell_w = float(img_size[1])/float(n_x_grids)
    cell_h = float(img_size[0])/float(n_y_grids)
    labels = np.zeros((n_x_grids*n_y_grids*n_anchors*n_elements_per_anchor, 1), dtype=np.float32)
    normalized_bbox = []
    norm_w_list = []
    norm_h_list = []
    for grid_x in range(n_x_grids):
        for grid_y in range(n_y_grids):
            x, y, w, h = 0, 0, 0, 0
            conf = 0.0
            anchors_found = [False]*n_anchors
            x_y_list = []
            w_h_list = []
            for bbox in bboxes_copy:
                width = float(bbox[2] - bbox[0])
                height = float(bbox[3] - bbox[1])
                center_x = float(bbox[0]) + (width/2.0)
                center_y = float(bbox[1]) + (height/2.0)
                x = center_x/cell_w
                y = center_y/cell_h

                if (int(math.floor(x)) == grid_x) and \
                   (int(math.floor(y)) == grid_y) :
                    w = width/float(img_size[1])
                    h = height/float(img_size[0])
                    x_y_list.append([x - float(grid_x), y - float(grid_y)])
                    w_h_list.append([w, h])

                    bboxes_copy.remove(bbox)

            if(len(x_y_list) > 0):
                anchor_idx = get_anchor_idx(w_h_list)
    
                for i in range(len(anchor_idx)):
                    if anchor_idx[i] != -1:
                        start_idx = (grid_x*n_y_grids+grid_y)*n_anchors*n_elements_per_anchor + (anchor_idx[i]*n_elements_per_anchor)
                        scaled_w = w_h_list[i][0]/anchors[anchor_idx[i]][0]
                        if(scaled_w == 0.):
                            scaled_w = 0.001
                        norm_w = math.log(scaled_w)/10.0 + 0.5
                        scaled_h = w_h_list[i][1]/anchors[anchor_idx[i]][1]
                        if(scaled_h == 0.):
                            scaled_h = 0.001
                        norm_h = math.log(scaled_h)/10.0 + 0.5
                        conf = 1.0

                        # label = [x_y_list[i][0], x_y_list[i][1], norm_w, norm_h, conf]
                        label = [x_y_list[i][0], x_y_list[i][1], w_h_list[i][0], w_h_list[i][1], conf]
                        
                        labels[start_idx:start_idx+n_elements_per_anchor, 0] = label

    return labels

# get bounding-boxes and confidence from assembled label
def disassemble_label(label, img_size) :
    cell_w = float(img_size[1]/n_x_grids)
    cell_h = float(img_size[0]/n_y_grids)
    bbox = []
    for grid_x in range(n_x_grids):
        for grid_y in range(n_y_grids):
            for n in range(n_anchors):
                start_idx = (grid_x*n_y_grids+grid_y)*n_anchors*n_elements_per_anchor + (n*n_elements_per_anchor)
                [x, y, norm_w, norm_h, conf] = label[start_idx:start_idx+n_elements_per_anchor]
                center_x = x*cell_w + (float(grid_x)*cell_w)
                center_y = y*cell_h + (float(grid_y)*cell_h)

                # width = (anchors[n][0]*math.exp((norm_w - 0.5)*10.0))*float(img_size[1])
                # height = (anchors[n][1]*math.exp((norm_h - 0.5)*10.0))*float(img_size[0])

                width = norm_w*float(img_size[1])
                height = norm_h*float(img_size[0])

                x_min = int(center_x - width/2)
                y_min = int(center_y - height/2)
                x_max = int(center_x + width/2)
                y_max = int(center_y + height/2)
                bbox += [x_min, y_min, x_max, y_max, conf]
    return bbox

# labels to bounding-boxes, and confidence
def labels2bboxes(labels, img_shape, nms=True):
    labels = disassemble_label(labels, [img_shape[0], img_shape[1]])
    bbox_list = []
    conf_list = []
    for grid_x in range(n_x_grids):
        for grid_y in range(n_y_grids):
            for n in range(n_anchors):
                start_idx = (grid_x*n_y_grids+grid_y)*n_anchors*n_elements_per_anchor + (n*n_elements_per_anchor)

                # get bbox
                bbox = labels[start_idx:start_idx+n_elements_per_anchor]
                conf = bbox[4]

                # append to list
                bbox_list.append(bbox[0:4])
                conf_list.append(conf)

    if(nms==True):
        bbox_list, conf_list = nonMaxSuppression(bbox_list, conf_list)

    # return
    return np.asarray(bbox_list).astype(np.int), np.asarray(conf_list)

# intersection-over-union
def iou(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

# non-max suppression
def nonMaxSuppression(bboxes, conf, conf_threshold=0.5, iou_threshold=0.5):
    bboxes_copy = bboxes.copy()
    conf_copy = conf.copy()

    if(len(bboxes_copy) > 0):
        # delete boxes with low confidence
        i=0
        while i < len(bboxes_copy):
            if(conf_copy[i] < conf_threshold):
                del bboxes_copy[i]
                del conf_copy[i]
                i-=1
            i+=1

        # delete boxes with high iou
        i,j = 0,0
        while i < len(bboxes_copy):
            while j < len(bboxes_copy):
                if i != j:
                    if(iou(bboxes_copy[i], bboxes_copy[j]) > iou_threshold):
                        if(conf_copy[i] > conf_copy[j]):
                            del bboxes_copy[j]
                            del conf_copy[j]
                            j-=1
                        else:
                            del bboxes_copy[i]
                            del conf_copy[i]
                            i-=1
                j+=1
            i+=1

    return bboxes_copy, conf_copy
