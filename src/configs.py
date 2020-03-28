import configparser

CONFIG_FILE = './config/config.ini'
# get configs
config = configparser.ConfigParser()
config.read(CONFIG_FILE)

# default data paths
data_path = config['PATHS']['DATA_PATH']
annotated_bbox2d_file = config['PATHS']['ANNOTATION_FILE']
camera_intrinsics_file = data_path + config['PATHS']['CAMERA_INTRINSICS_FILE']
anchor_list_file = config['PATHS']['ANCHOR_LIST_FILE']
labels_list_file = config['PATHS']['LABELS_LIST_FILE']

# original image dimension
original_img_w = config.getint('IMG_DIM','WIDTH',fallback=1242)
original_img_h = config.getint('IMG_DIM','HEIGHT',fallback=375)

# resize shape
resize_img_w = config.getint('IMG_RESIZE_DIM','WIDTH',fallback=256)
resize_img_h = config.getint('IMG_RESIZE_DIM','HEIGHT',fallback=256)
RESIZE_SHAPE = (resize_img_w, resize_img_h)

# hyperparameters
experiment_id = config['HYPERPARAMETERS']['EXPERIMENT_ID']
batch_size = config.getint('HYPERPARAMETERS','BATCH_SIZE',fallback=8)
epochs = config.getint('HYPERPARAMETERS','EPOCHS',fallback=101)
initial_epoch = config.getint('HYPERPARAMETERS','INITIAL_EPOCH',fallback=0)
learning_rate = config.getfloat('HYPERPARAMETERS','LEARNING_RATE',fallback=0.001)
beta_1 = config.getfloat('HYPERPARAMETERS','BETA_1',fallback=0.9)
beta_2 = config.getfloat('HYPERPARAMETERS','BETA_2',fallback=0.999)
decay = config.getfloat('HYPERPARAMETERS','DECAY',fallback=(learning_rate/epochs))

# Other config parameters
USE_GENERATOR = config.getboolean('HYPERPARAMETERS','USE_GENERATOR',fallback=True)
USE_PRETRAINED_WEIGHTS = config.getboolean('HYPERPARAMETERS','USE_PRETRAINED_WEIGHTS',fallback=True)
CLASS_PREDICTION = config.getboolean('HYPERPARAMETERS','CLASS_PREDICTION',fallback=True)
N_TRAINABLE_LAYERS_TAIL = config.getint('HYPERPARAMETERS','N_TRAINABLE_LAYERS_TAIL',fallback=-1)
N_LAYERS_LOAD_WEIGHTS_TAIL = config.getint('HYPERPARAMETERS','N_LAYERS_LOAD_WEIGHTS_TAIL',fallback=-1)
weight_backup_epoch = config.getint('HYPERPARAMETERS','WEIGHT_BACKUP_EPOCHS',fallback=10)
original_img_w = config.getint('IMG_DIM','WIDTH',fallback=1242)
original_img_h = config.getint('IMG_DIM','HEIGHT',fallback=375)
resize_img_w = config.getint('IMG_RESIZE_DIM','WIDTH',fallback=128)
resize_img_h = config.getint('IMG_RESIZE_DIM','HEIGHT',fallback=128)
weight_backup_epoch = config.getint('HYPERPARAMETERS','WEIGHT_BACKUP_EPOCHS',fallback=10)

# paths
WEIGHTS_FOLDER = config['PATHS']['WEIGHTS_FOLDER'] + experiment_id + '/'
WEIGHTS_FILE_basename = WEIGHTS_FOLDER + config['PATHS']['WEIGHTS_FILE_basename']
MODELS_FOLDER = config['PATHS']['MODELS_FOLDER'] + experiment_id + '/'
MODEL_FILE_basename = MODELS_FOLDER + config['PATHS']['MODEL_FILE_basename']
LOGS_FOLDER = config['PATHS']['LOGS_FOLDER'] 
CHECKPOINTS_FOLDER = config['PATHS']['CHECKPOINTS_FOLDER'] + experiment_id + '/'
CHECKPOINTS_FILE = CHECKPOINTS_FOLDER + config['PATHS']['CHECKPOINTS_FILE']
CSV_LOG_FILE = MODELS_FOLDER + config['PATHS']['CSV_LOG_FILE']
PRETRAINED_WEIGHT_FILE = config['PATHS']['PRETRAINED_WEIGHT_FILE']
INTERMEDIATE_RES_FOLDER = MODELS_FOLDER+'/'+config['PATHS']['INTERMEDIATE_RES_FOLDER']
if PRETRAINED_WEIGHT_FILE is 'NONE':
    USE_PRETRAINED_WEIGHTS = False

anchor_list_file = config['PATHS']['ANCHOR_LIST_FILE']

# grid
n_x_grids = config.getint('HYPERPARAMETERS','N_X_GRIDS',fallback=1)
n_y_grids = config.getint('HYPERPARAMETERS','N_Y_GRIDS',fallback=1)

# number of predicted bboxes, anchors, and elements per grid
n_bbox_per_grid = config.getint('HYPERPARAMETERS','N_BBOX_PER_GRID',fallback=2)
n_anchors = config.getint('HYPERPARAMETERS','N_ANCHORS',fallback=5)
n_classes = 7
n_elements_per_anchor = 1+4+n_classes # 1 conf, 4 bbox

# lambda
lambda_xy = config.getfloat('HYPERPARAMETERS','LAMBDA_XY',fallback=1.0)
lambda_wh = config.getfloat('HYPERPARAMETERS','LAMBDA_WH',fallback=1.0)
lambda_iou = config.getfloat('HYPERPARAMETERS','LAMBDA_IOU',fallback=1.0)
lambda_conf = config.getfloat('HYPERPARAMETERS','LAMBDA_CONF',fallback=1.0)
lambda_class = config.getfloat('HYPERPARAMETERS','LAMBDA_CLASS',fallback=1.0)

if CLASS_PREDICTION == False:
    n_elements_per_anchor = 1+4
    lambda_class = 0