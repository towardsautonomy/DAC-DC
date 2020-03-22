from keras.optimizers import Adam
from keras.utils import plot_model
from keras.backend.tensorflow_backend import set_session

from src.dataLoader import *
from src.model import *
from src.keras_utils import *
from src.LossFunc import *
from src.plotMetrics import plot_metrics
from testModel import testModel

if __name__ == '__main__':
    # load data
    n_samples = -1
    n_train_positive, n_test_positive = getNumSamples(n_samples=n_samples)
    steps_per_epoch_train = n_train_positive/batch_size
    steps_per_epoch_validation = n_test_positive/batch_size
    dataset = 'vkitti'

    # Disable GPU memory pre-allocation
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    #config.log_device_placement=True
    session = tf.Session(config=config)
    set_session(session)

    X_train, y_train, X_test, y_test = None, None, None, None
    train_gen, valid_gen = None, None
    if USE_GENERATOR == False:
        X_train, y_train, X_test, y_test = getData(n_samples=n_samples, dim=(resize_img_w, resize_img_h), dataset=dataset)
    else:
        train_gen, valid_gen = getDataGenerator(n_samples=n_samples, dim=(resize_img_w, resize_img_h), dataset=dataset)

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
                            metrics=[xy_loss, wh_loss, conf_loss, iou_loss, total_loss, mean_iou])

    # check whether to load weights from a pretrained weight file
    if USE_PRETRAINED_WEIGHTS == True:
        parallel_model.load_weights(PRETRAINED_WEIGHT_FILE)
        # set the trainability of model
        if N_TRAINABLE_LAYERS_TAIL is not -1:
            for layer in model.layers[:-N_TRAINABLE_LAYERS_TAIL]:
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
                                                callbacks=[tensorboard, checkpoint, early_stop, WeightsSaver(parallel_model, weight_backup_epoch), csv_logger])
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
