from keras import backend as K
import tensorflow as tf
import configparser

from src.configs import *

# Cross Entropy
def CrossEntropy(y_true, y_pred):
    return -K.sum(y_true*K.log(K.maximum(y_pred, 0.00001)) + (1.0-y_true)*K.log(K.maximum((1.0-y_pred), 0.00001)))

# xy loss
def xy_loss(y_true, y_pred):
    # reshape
    y_true = K.reshape(y_true, shape=(-1,n_x_grids*n_y_grids,n_anchors,n_elements_per_anchor))
    y_pred = K.reshape(y_pred, shape=(-1,n_x_grids*n_y_grids,n_anchors,n_elements_per_anchor))

    # x and y for prediction
    y_pred_xy   = y_pred[...,1:3]

    # x and y for ground-truth
    y_true_xy   = y_true[...,1:3]

    # probability that there is something in that cell.
    y_true_conf = y_true[...,0]

    # compute xy loss
    xy_loss    = K.sum(K.sum(K.square(y_true_xy - y_pred_xy),axis=-1)*y_true_conf, axis=-1)
    return (lambda_xy * xy_loss)

# wh loss
def wh_loss(y_true, y_pred):
    # reshape
    y_true = K.reshape(y_true, shape=(-1,n_x_grids*n_y_grids,n_anchors,n_elements_per_anchor))
    y_pred = K.reshape(y_pred, shape=(-1,n_x_grids*n_y_grids,n_anchors,n_elements_per_anchor))

    # w and h predicted are 0 to 1 with 1 being image size
    y_pred_wh   = y_pred[...,3:5]

    # width and height for ground-truth
    y_true_wh   = y_true[...,3:5]

    # probability that there is something in that cell.
    y_true_conf = y_true[...,0]

    # compute wh loss
    wh_loss = K.sum(K.sum(K.square(K.sqrt(y_true_wh) - K.sqrt(y_pred_wh)), axis=-1)*y_true_conf, axis=-1)
    return (lambda_wh * wh_loss)

# iou
def iou(y_true, y_pred):
    # reshape
    y_true = K.reshape(y_true, shape=(-1,n_x_grids*n_y_grids,n_anchors,n_elements_per_anchor))
    y_pred = K.reshape(y_pred, shape=(-1,n_x_grids*n_y_grids,n_anchors,n_elements_per_anchor))

    # x and y for prediction
    y_pred_xy   = y_pred[...,1:3]
    # w and h predicted are 0 to 1 with 1 being image size
    y_pred_wh   = y_pred[...,3:5]

    # x and y for ground-truth
    y_true_xy   = y_true[...,1:3]
    # width and height
    y_true_wh   = y_true[...,3:5]

    # compute iou for all boxes at once 
    intersect_wh = K.maximum(K.zeros_like(y_pred_wh), (y_pred_wh + y_true_wh)/2 - K.square(y_pred_xy - y_true_xy) )
    intersect_area = intersect_wh[...,0] * intersect_wh[...,1]
    true_area = y_true_wh[...,0] * y_true_wh[...,1]
    pred_area = y_pred_wh[...,0] * y_pred_wh[...,1]
    union_area = pred_area + true_area - intersect_area
    iou = (intersect_area + 1e-5) / (union_area + 1e-5)

    return iou

# mean iou
def mean_iou(y_true, y_pred):
    # get iou
    iou_ = iou(y_true, y_pred)

    # reshape
    y_true = K.reshape(y_true, shape=(-1,n_x_grids*n_y_grids,n_anchors,n_elements_per_anchor))

    # ground-truth confidence
    y_true_conf = y_true[...,0]

    # number of true bounding-boxes
    n_true_bbox = K.sum(K.sum(y_true_conf, axis=-1))

    mean_iou_ = K.sum(K.sum((y_true_conf*iou_), axis=-1)) / (n_true_bbox + 1e-5)

    return mean_iou_

# iou loss
def iou_loss(y_true, y_pred):
    # reshape
    y_true = K.reshape(y_true, shape=(-1,n_x_grids*n_y_grids,n_anchors,n_elements_per_anchor))

    # ground-truth confidence
    y_true_conf = y_true[...,0]
    iou_ = iou(y_true, y_pred)

    iou_loss_ = -K.sum(y_true_conf*K.log(K.maximum(iou_, 0.00001)), axis=-1)

    return (lambda_iou * iou_loss_)

# confidence loss
def conf_loss(y_true, y_pred):
    # reshape
    y_true = K.reshape(y_true, shape=(-1,n_x_grids*n_y_grids,n_anchors,n_elements_per_anchor))
    y_pred = K.reshape(y_pred, shape=(-1,n_x_grids*n_y_grids,n_anchors,n_elements_per_anchor))

    # predicted confidence
    y_pred_conf = y_pred[...,0]

    # ground-truth confidence
    y_true_conf = y_true[...,0]

    # iou
    iou_ = iou(y_true, y_pred)

    # compute loss
    # conf_loss = K.sum(K.sum(K.square(y_true_conf*iou_ - y_pred_conf), axis=-1))
    # conf_loss = K.mean(K.sum(K.square(y_true_conf*iou_ - y_pred_conf)*y_true_conf, axis=-1)+
    #                    K.sum(K.square(y_true_conf - y_pred_conf)*(1. - y_true_conf), axis=-1))
    conf_loss = K.mean(K.sum(K.square(y_true_conf - y_pred_conf)*y_true_conf, axis=-1)+
                       K.sum(K.square(y_true_conf - y_pred_conf)*(1. - y_true_conf), axis=-1))

    return (lambda_conf * conf_loss)

# class loss
def class_loss(y_true, y_pred):
    # reshape
    y_true = K.reshape(y_true, shape=(-1,n_x_grids*n_y_grids,n_anchors,n_elements_per_anchor))
    y_pred = K.reshape(y_pred, shape=(-1,n_x_grids*n_y_grids,n_anchors,n_elements_per_anchor))

    # ground-truth confidence
    y_true_conf = y_true[...,0]

    # true class
    y_true_class = y_true[...,5:]

    # predicted class
    y_pred_class = y_pred[...,5:]

    # compute loss
    class_loss = K.sum(K.mean(K.sum(K.square(y_true_class - y_pred_class), axis=-1)*y_true_conf))

    return (lambda_class * class_loss)

# total loss
def total_loss(y_true, y_pred):
    # reshape
    y_true = K.reshape(y_true, shape=(-1,n_x_grids*n_y_grids,n_anchors,n_elements_per_anchor))

    # number of true bounding-boxes
    n_true_bbox = K.sum(K.sum(y_true[...,0], axis=-1))

    # n_false_boxes = K.cast(K.tf.count_zero(y_true[...,0]), K.tf.float32)

    # get respective losses
    xy_loss_ = xy_loss(y_true, y_pred)/(n_true_bbox + 1e-5)
    wh_loss_ = wh_loss(y_true, y_pred)/(n_true_bbox + 1e-5)
    iou_loss_ = iou_loss(y_true, y_pred)/(n_true_bbox + 1e-5)
    conf_loss_ = conf_loss(y_true, y_pred)
    class_loss_ = class_loss(y_true, y_pred)/(n_true_bbox + 1e-5)

    # compute total loss
    total_loss_ = xy_loss_    + \
                  wh_loss_    + \
                  iou_loss_   + \
                  conf_loss_  + \
                  class_loss_

    # total_loss_ = xy_loss_    + \
    #               wh_loss_    + \
    #               conf_loss_

    return total_loss_

# loss function
def LossFunc_2DOD(y_true, y_pred):
    return total_loss(y_true, y_pred)