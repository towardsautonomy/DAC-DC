from keras.optimizers import Adam
from keras.utils import plot_model
from keras.backend.tensorflow_backend import set_session

from src.model import *
from src.keras_utils import *
from src.LossFunc import *
from src.DACDC_DataLoader import DACDC_DataLoader
from src.plotMetrics import plot_metrics
from testModel import testModel

if __name__ == '__main__':
    # Disable GPU memory pre-allocation
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    #config.log_device_placement=True
    session = tf.Session(config=config)
    set_session(session)

    # data loader
    dacdc_dloader = DACDC_DataLoader( data_path, annotated_bbox2d_file)

    n_samples = -1
    n_train_positive, n_test_positive = dacdc_dloader.get_num_samples(n_samples=n_samples)
    steps_per_epoch_train = n_train_positive/batch_size
    steps_per_epoch_validation = n_test_positive/batch_size

    X_train, y_train, X_test, y_test = None, None, None, None
    train_gen, valid_gen = None, None
    if USE_GENERATOR == False:
        X_train, y_train, X_test, y_test = dacdc_dloader.get_data(n_samples=n_samples, dim=(resize_img_w, resize_img_h))
    else:
        train_gen, valid_gen = dacdc_dloader.get_data_generator(n_samples=n_samples, batch_size=batch_size, dim=(resize_img_w, resize_img_h))

    model = None
    parallel_model = None
    with tf.device('/gpu:0'):
        # load model
        model = DAC_DC(shape=(resize_img_h,resize_img_w,3))
        plot_model(model, to_file=MODEL_FILE_basename+'.png', show_shapes=True, show_layer_names=True)

    # create multi-gpu model if available
    try:
        parallel_model = ModelMGPU(model, get_n_gpu())
        print('Using {} GPUs for training.'.format(get_n_gpu()))
    except:
        parallel_model = model

    # define optimizer and compile model
    optimizer = Adam(lr=learning_rate, beta_1=beta_1, beta_2=beta_2, decay=decay)

    parallel_model.compile( loss=LossFunc_2DOD,
                            optimizer=optimizer,
                            metrics=[xy_loss, wh_loss, conf_loss, class_loss, iou_loss, total_loss, mean_iou])

    # check whether to load weights from a pretrained weight file
    if USE_PRETRAINED_WEIGHTS == True:
        if N_LAYERS_LOAD_WEIGHTS_TAIL is not -1:
            pretrained_model = None
            with tf.device('/cpu:0'):
                pretrained_model = load_model(PRETRAINED_WEIGHT_FILE, custom_objects={'LossFunc_2DOD'  : LossFunc_2DOD,
                                                                                      'xy_loss'        : xy_loss,
                                                                                      'wh_loss'        : wh_loss,
                                                                                      'conf_loss'      : conf_loss,
                                                                                      'class_loss'     : class_loss,
                                                                                      'iou_loss'       : iou_loss,
                                                                                      'total_loss'     : total_loss,
                                                                                      'mean_iou'       : mean_iou})

            print('Loading weights from pretrained model [{}] into current model'.format(PRETRAINED_WEIGHT_FILE))
            for i, layer in enumerate(parallel_model.layers[:-N_LAYERS_LOAD_WEIGHTS_TAIL]):
                layer.set_weights(pretrained_model.layers[i].get_weights())
                print('[{}] Loaded weights from  layer [{}] into layer [{}] of current model'.format(i, pretrained_model.layers[i].name, layer.name))
            del pretrained_model
        else:
            parallel_model.load_weights(PRETRAINED_WEIGHT_FILE)
            print('Loaded weights from pretrained model [{}] into current model'.format(PRETRAINED_WEIGHT_FILE))
        # set the trainability of model
        if N_TRAINABLE_LAYERS_TAIL is not -1:
            for layer in parallel_model.layers[:-N_TRAINABLE_LAYERS_TAIL]:
                layer.trainable = False

    # summarize parameters
    trainable_count = int(
        np.sum([K.count_params(p) for p in set(parallel_model.trainable_weights)]))
    non_trainable_count = int(
        np.sum([K.count_params(p) for p in set(parallel_model.non_trainable_weights)]))

    print("=============================================================")
    print('Total params: {:,}'.format(trainable_count + non_trainable_count))
    print('Trainable params: {:,}'.format(trainable_count))
    print('Non-trainable params: {:,}'.format(non_trainable_count))
    print("=============================================================")

    # fit model
    if USE_GENERATOR == False:
        history = parallel_model.fit(           X_train, y_train, 
                                                batch_size=batch_size, 
                                                epochs=epochs, 
                                                shuffle=True,
                                                verbose=1, 
                                                initial_epoch=initial_epoch,
                                                validation_data=(X_test, y_test),
                                                callbacks=[tensorboard, checkpoint, early_stop, WeightsSaver(parallel_model, weight_backup_epoch), csv_logger, epoch_end_test])
    else:
        history = parallel_model.fit_generator( train_gen, 
                                                steps_per_epoch=steps_per_epoch_train,
                                                validation_data=valid_gen, 
                                                validation_steps=steps_per_epoch_validation,
                                                use_multiprocessing=True,
                                                workers=4,
                                                shuffle=True, 
                                                epochs=epochs, 
                                                verbose=1,
                                                initial_epoch=initial_epoch,
                                                callbacks=[tensorboard, checkpoint, early_stop, WeightsSaver(parallel_model, weight_backup_epoch), csv_logger])

    # save model
    parallel_model.save(MODEL_FILE_basename+'.h5')
    print('Saved the trained model to : {}'.format(MODEL_FILE_basename+'.h5'))

    # plot metrics
    plot_metrics()

    # test model
    testModel(MODEL_FILE_basename+'.h5', dataset=dataset)
