from tensorflow.python.client import device_lib
from keras.utils import multi_gpu_model
from keras.models import Model
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, TensorBoard
from keras.callbacks import CSVLogger
import os
from time import time
import configparser
import cv2
import io
import numpy as np
import matplotlib.pyplot as plt

from src.configs import *
from src.dacdc_utils import *

# create folders
os.system('mkdir -p '+str(WEIGHTS_FOLDER))
os.system('mkdir -p '+str(MODELS_FOLDER))
os.system('mkdir -p '+str(CHECKPOINTS_FOLDER))
os.system('mkdir -p '+str(LOGS_FOLDER))
os.system('mkdir -p '+str(INTERMEDIATE_RES_FOLDER))

# copy configs to the experiment folder
os.system('cp '+CONFIG_FILE+' '+MODELS_FOLDER)
os.system('cp '+anchor_list_file+' '+MODELS_FOLDER)

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def get_n_gpu():
    return len(get_available_gpus())

def textOnImage(image, position, text, color=(0,0,0), fontScale = 0.7, thickness = 1):
    font                   = cv2.FONT_HERSHEY_SIMPLEX    
    bottomLeftCornerOfText = position
    fontColor              = color
    lineType               = 2

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

class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)

class LossAndErrorPrintingCallback(Callback):
    def on_train_batch_end(self, batch, logs=None):
        print('For batch {}, loss is {:7.2f}.'.format(batch, logs['loss']))

    def on_test_batch_end(self, batch, logs=None):
        print('For batch {}, loss is {:7.2f}.'.format(batch, logs['loss']))

    def on_epoch_end(self, epoch, logs=None):
        print('The average loss for epoch {} is {:7.2f}.'.format(epoch, logs['loss']))

class EpochEndTestCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        test_img_idx = 0
        n_samples = 8
        y_pred = self.model.predict(self.validation_data[0][0:n_samples])
        y_true = self.validation_data[1][0:n_samples]

        # cv2.imwrite('epoch_test/img.png', img)
        # print('prediction: {}'.format(y_pred[0]))

        # grid subplots
        n_cols = 4
        n_rows = 2
        plt.figure(figsize=(32,18))
        for i in range(n_rows):
            for j in range(n_cols):
                idx = i*n_cols+j
                plt.subplot(n_rows, n_cols, idx+1)
                # retrieve label
                img_curr = np.asarray(np.multiply((self.validation_data[0][idx] + 0.5), 255.0), dtype=np.int32)

                conf_pred, bboxes_pred, classes_pred = labels2bboxes(y_pred[idx], img_size=[img_curr.shape[0], img_curr.shape[1]])
                conf_true, bboxes_true, classes_true = labels2bboxes(y_true[idx], img_size=[img_curr.shape[0], img_curr.shape[1]],  nms=False)

                for k, bbox in enumerate(bboxes_true):
                    # draw bbox if conf is > 50%
                    if(conf_true[k] > 0.5):
                        img_curr = cv2.rectangle(img_curr, (bbox[0],bbox[1]), (bbox[2],bbox[3]), color=(0,0,255), thickness=2)

                for k, bbox in enumerate(bboxes_pred):
                    # draw bbox if conf is > 50%
                    if(conf_pred[k] > 0.5):
                        img_curr = cv2.rectangle(img_curr, (bbox[0],bbox[1]), (bbox[2],bbox[3]), color=(255,0,0), thickness=2)
                        img_curr = cv2.rectangle(img_curr, (bbox[0],bbox[1]-15), (bbox[0]+75,bbox[1]-2), color=(0,0,0), thickness=-1)
                        if CLASS_PREDICTION == True:
                            textOnImage(img_curr, (bbox[0], bbox[1]-5), classes_true[k]+' {:.1f}%'.format(conf_pred[k]*100.0), color=(255,255,255), fontScale=0.3, thickness=1)
                        else:
                            textOnImage(img_curr, (bbox[0], bbox[1]-5), 'Conf {:.1f}%'.format(conf_pred[k]*100.0), color=(255,255,255), fontScale=0.3, thickness=1)

                # show result
                plt.imshow(img_curr)

        plt.savefig(INTERMEDIATE_RES_FOLDER+'/{}.png'.format(str(epoch).zfill(4)))
        plt.close()

# define weight saver callback
class WeightsSaver(Callback):
    def __init__(self, model, N):
        self.model = model
        self.N = N
        self.epoch = 0

    def on_epoch_end(self, epoch, logs={}):
        if (self.epoch > 0) and (self.epoch % self.N == 0):
            name = WEIGHTS_FILE_basename + '_%04d.weights' % (self.epoch)
            self.model.save_weights(name)
        self.epoch += 1

batch_loss_info = LossAndErrorPrintingCallback()
epoch_end_test = EpochEndTestCallback()

# define checkpoint callback
checkpoint = ModelCheckpoint(CHECKPOINTS_FILE, monitor='loss', verbose=1, save_best_only=True)

# define early stop callback
early_stop = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)

# define tensorboard callback
tensorboard = TensorBoard(log_dir=LOGS_FOLDER+'{}'.format(time()))

# define CSV Logger callback
csv_logger = CSVLogger(CSV_LOG_FILE, append=True, separator=',')
