from tensorflow.python.client import device_lib
from keras.utils import multi_gpu_model
from keras.models import Model
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, TensorBoard
from keras.callbacks import CSVLogger
import os
from time import time
import configparser

from src.configs import *

# create folders
os.system('mkdir -p '+str(WEIGHTS_FOLDER))
os.system('mkdir -p '+str(MODELS_FOLDER))
os.system('mkdir -p '+str(CHECKPOINTS_FOLDER))
os.system('mkdir -p '+str(LOGS_FOLDER))

# copy configs to the experiment folder
os.system('cp '+CONFIG_FILE+' '+MODELS_FOLDER)
os.system('cp '+anchor_list_file+' '+MODELS_FOLDER)

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def get_n_gpu():
    return len(get_available_gpus())

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

# define checkpoint callback
checkpoint = ModelCheckpoint(CHECKPOINTS_FILE, monitor='loss', verbose=1, save_best_only=True)

# define early stop callback
early_stop = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)

# define tensorboard callback
tensorboard = TensorBoard(log_dir=LOGS_FOLDER+'{}'.format(time()))

# define CSV Logger callback
csv_logger = CSVLogger(CSV_LOG_FILE, append=True, separator=',')
