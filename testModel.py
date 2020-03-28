from keras.models import load_model
from keras.backend.tensorflow_backend import set_session
import math

from src.DACDC_DataLoader import DACDC_DataLoader
from src.model import *
from src.keras_utils import *
from src.dacdc_utils import *
from src.LossFunc import *

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

def testModel(model_file, dataset='vkitti'):
    print('-------------------------------')
    print('Testing the trained model')
    print('-------------------------------')
    # load model
    model = load_model(model_file, custom_objects={'LossFunc_2DOD'  : LossFunc_2DOD,
                                                   'xy_loss'        : xy_loss,
                                                   'wh_loss'        : wh_loss,
                                                   'conf_loss'      : conf_loss,
                                                   'class_loss'     : class_loss,
                                                   'iou_loss'       : iou_loss,
                                                   'total_loss'     : total_loss,
                                                   'mean_iou'       : mean_iou})
    print('Model Loaded')

    # load data
    dacdc_dloader = DACDC_DataLoader( data_path, annotated_bbox2d_file)
    x_train, y_train, x_test, y_test = dacdc_dloader.get_data(n_samples=32, test_size=1.0, dim=(resize_img_w, resize_img_h))
    print('Data Loaded')

    # number of samples to perform test on
    n_samples_test = 8
    test_imgs = x_test[0:n_samples_test]

    # perform prediction
    y_pred = np.array(model.predict(test_imgs))
    y_true = np.array(y_test[0:n_samples_test])
    
    # plot result
    img = dacdc_dloader.denormalize_img(x_test)

    # grid subplots
    n_cols = 4
    n_rows = math.ceil(n_samples_test/n_cols)
    plt.figure(figsize=(32,18))
    for i in range(n_rows):
        for j in range(n_cols):
            idx = i*n_cols+j
            plt.subplot(n_rows, n_cols, idx+1)
            # retrieve label
            img_curr = img[idx]

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

    # save and show the figure
    plt.savefig(MODELS_FOLDER+'result.png')
    plt.show()

if __name__ == '__main__':
    # Disable GPU memory pre-allocation
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    #config.log_device_placement=True
    session = tf.Session(config=config)
    set_session(session)

    testModel(PRETRAINED_WEIGHT_FILE, dataset='vkitti')