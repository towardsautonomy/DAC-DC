from keras.models import load_model
from keras.backend.tensorflow_backend import set_session
import math
import glob
import os

from src.dataLoader import *
from src.model import *
from src.dacdc_utils import *
from src.LossFunc import *

import time

data_root = '/home/shubham/workspace/dataset/vKITTI/'
data_path = data_root+'*/*/frames/rgb/Camera_0/*.jpg'
dst_path = '/home/shubham/workspace/dataset/vKITTI_2dod/'

'''
data_root = '/media/shubham/Shubz/workspace/dataset/KITTI/'
data_path = data_root+'dataset/sequences/00/image_2/*.png'
dst_path = '/media/shubham/Shubz/workspace/dataset/KITTI_2dod/'
'''

model_file = '/home/shubham/workspace/SingleShot_2DOD/pretrained_weights/vkitti.h5'
resize_dim = (256,256)

write_to_file = False

# Disable GPU memory pre-allocation
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
#config.log_device_placement=True
session = tf.Session(config=config)
set_session(session)

def textOnImage(image, position, text, color=(0,0,0), fontScale = 0.7, thickness = 1):
    font                   = cv2.FONT_HERSHEY_COMPLEX_SMALL  
    bottomLeftCornerOfText = position
    fontColor              = color
    lineType               = cv2.LINE_AA

    dy = 40
    for i, line in enumerate(text.split('\n')):
        y = bottomLeftCornerOfText[1] + i*dy
        cv2.putText(image,line, 
            (bottomLeftCornerOfText[0], y), 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)

if __name__ == '__main__':
    # load model
    model = load_model(model_file, custom_objects={ 'LossFunc_2DOD'  : LossFunc_2DOD,
                                                    'xy_loss'        : xy_loss,
                                                    'wh_loss'        : wh_loss,
                                                    'conf_loss'      : conf_loss,
                                                    'iou_loss'       : iou_loss,
                                                    'total_loss'     : total_loss,
                                                    'mean_iou'       : mean_iou})
    print('Model Loaded')

    # load data
    fnames = sorted(glob.glob(data_path))

    for i in range(len(fnames)):
        # read, resize, and normalize image
        img = cv2.cvtColor(cv2.imread(fnames[i]), cv2.COLOR_BGR2RGB)
        img_size = [img.shape[0], img.shape[1]]
        img_resized = cv2.resize(img, resize_dim)
        img_norm = np.asarray(normalize_img(img_resized), dtype=np.float32)
        img_norm = img_norm[np.newaxis,...]

        # perform prediction
        y_pred = np.array(model.predict(img_norm))

        bboxes, conf = labels2bboxes(y_pred[0], [img.shape[0], img.shape[1]])

        for j, bbox in enumerate(bboxes):
            # draw bbox if conf is > 50%
            if(conf[j] > 0.5):
                img = cv2.rectangle(img, (bbox[0],bbox[1]), (bbox[2],bbox[3]), color=(255,0,0), thickness=2)
                img = cv2.rectangle(img, (bbox[0],bbox[1]-15), (bbox[0]+75,bbox[1]-2), color=(0,0,0), thickness=-1)
                textOnImage(img, (bbox[0], bbox[1]-5), 'Conf: {:.1f}%'.format(conf[j]*100.0), color=(255,255,255), fontScale=0.5, thickness=1)
        
        # rgb to bgr
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if(write_to_file == True):
            # write to file
            dst_dir = dst_path+fnames[i][len(data_root):-13]
            if(not os.path.exists(dst_dir)):
                os.system('mkdir -p '+dst_dir)
            dst_fname = dst_path+fnames[i][len(data_root):]

            cv2.imwrite(dst_fname, bgr)
            print('wrote -> {}'.format(dst_fname))

        else:
            cv2.imshow('detections', bgr)
            cv2.waitKey(50)