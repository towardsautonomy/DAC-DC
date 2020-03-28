import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import statistics
import math
import glob

from src.dacdc_utils import *

# normalize image
def normalize_img(X):
    return (np.true_divide(X, 255.0) - 0.5)

# denormalize image
def denormalize_img(X):
    return np.asarray(np.multiply((X + 0.5), 255.0), dtype=np.int32)

# assemble data from vkitti dataset
def assemble_data_vkitti(n_samples=-1, test_size=0.1, data_path=data_path):
    annotations = glob.glob(data_path+'*/*/'+annotated_bbox2d_file)

    fnames_list = []
    annotations_list = []
    flags_list = []
    fnames_train = []
    annotations_train = []
    flags_train = []
    fnames_test = []
    annotations_test = []
    flags_test = []

    for annotation in annotations:
        fnames = []
        bbox2d_exists = []

        # Open annotated 2D bounding-boxes
        with open(annotation, 'r') as file:
            bbox2d_str = file.readlines()

        # Set up a list in which each element specifies
        # if bounding box for that file index exists
        bbox2d_last_idx = int(bbox2d_str[len(bbox2d_str)-1].replace('\n', '').split()[0])
        bbox2d_list = [[] for i in range(bbox2d_last_idx+1)] 
        bbox2d_exists = [False for i in range(bbox2d_last_idx+1)] 
        fnames = ['' for i in range(bbox2d_last_idx+1)] 

        for i in range(1,len(bbox2d_str)):
            line = bbox2d_str[i].replace('\n', '').split()
            bbox_list = []
            if(int(line[1]) == 0): # camera 0
                bbox2d_exists[int(line[0])] = True
                bbox_list.append([float(j) for j in line[3:7]])
            for bbox in bbox_list:
                [xmin, xmax, ymin, ymax] = bbox
                bbox2d_list[int(line[0])].append([xmin, ymin, xmax, ymax])
            
            fnames[int(line[0])] = annotation[:-8]+'frames/rgb/Camera_0/rgb_' + str(line[0]).zfill(5) + '.jpg'

        # convert to numpy array
        fnames = np.asarray(fnames)
        annotations = np.asarray(bbox2d_list)
        bbox2d_exists = np.asarray(bbox2d_exists)

        # add to the list
        fnames_list.extend(fnames)
        annotations_list.extend(annotations)
        flags_list.extend(bbox2d_exists)

    # shuffle the data
    assert len(fnames_list) == len(annotations_list)
    assert len(fnames_list) == len(flags_list)
    p = np.random.permutation(len(fnames_list))
    fnames_list = np.asarray(fnames_list)[p]
    annotations_list = np.asarray(annotations_list)[p]
    flags_list = np.asarray(flags_list)[p]

    # compute number of train and test samples
    if(n_samples >= len(fnames_list)) or (n_samples == -1): 
        n_samples = len(fnames_list)

    n_test = int(float(n_samples)*test_size)
    n_train = n_samples - n_test

    ## Split train and test samples
    # filenames
    fnames_train = fnames_list[0:n_train]
    fnames_test = fnames_list[n_train:n_train+n_test]
    # annotations
    annotations_train = annotations_list[0:n_train]
    annotations_test = annotations_list[n_train:n_train+n_test]
    # flags
    flags_train = flags_list[0:n_train]
    flags_test = flags_list[n_train:n_train+n_test]
    # return
    return fnames_train, annotations_train, flags_train, fnames_test, annotations_test, flags_test

# assemble data (format: frame_num num_bbox xmin1 ymin1 xmax1 ymax2 xmin2 ymax2 ... )
def assemble_data_bbox_vector(n_samples=-1, test_size=0.1, data_path=data_path):
    bbox2d_str = None
    fnames = []
    annotations = []
    bbox2d_exists = []

    # Open annotated 2D bounding-boxes
    with open(data_path+annotated_bbox2d_file, 'r') as file:
        bbox2d_str = file.readlines()
    
    # Set up a list in which each element specifies
    # if bounding box for that file index exists
    bbox2d_last_idx = int(bbox2d_str[len(bbox2d_str)-1].replace('\n', '').split()[0])
    bbox2d_exists = [False for i in range(bbox2d_last_idx+1)] 

    # Create an ordered list of strings where the each line can
    # be indexed by file index
    bbox2d_str_ordered = [' ' for i in range(bbox2d_last_idx+1)]
    for i in range(len(bbox2d_str)):
        line = bbox2d_str[i].replace('\n', '').split()
        if(int(line[1]) > 0):
            bbox2d_str_ordered[int(line[0])] = line
            bbox2d_exists[int(line[0])] = True

    # Iterate through all the bounding-boxes
    for i in range(bbox2d_last_idx+1):
        # list of 2d bounding boxes
        bbox2d_list = []
        n_2d_bbox = 0
        if(len(bbox2d_str_ordered[i]) > 1):
            n_2d_bbox = int(bbox2d_str_ordered[i][1])
        
        # get a list of 2d bounding-boxes
        for j in range(n_2d_bbox):
            bbox2d_list.append([float(bbox2d_str_ordered[i][2 + 4*j]),
                                float(bbox2d_str_ordered[i][3 + 4*j]),
                                float(bbox2d_str_ordered[i][4 + 4*j]),
                                float(bbox2d_str_ordered[i][5 + 4*j])])

        # read corresponding image
        img_fname = data_path + 'rgb/frame_' + str(i).zfill(4) + '.png'

        # append to the list of file names and annotations
        fnames.append(img_fname)
        annotations.append(bbox2d_list)

    # convert to numpy array
    fnames = np.asarray(fnames)
    annotations = np.asarray(annotations)
    bbox2d_exists = np.asarray(bbox2d_exists)

    # shuffle the data
    assert len(fnames) == len(annotations)
    assert len(fnames) == len(bbox2d_exists)
    p = np.random.permutation(len(fnames))
    fnames = fnames[p]
    annotations = annotations[p]
    bbox2d_exists = bbox2d_exists[p]

    # for i in range(bbox2d_last_idx+1):
    #     print(fnames[i], annotations[i], bbox2d_exists[i])

    # compute number of train and test samples
    if(n_samples >= len(fnames)) or (n_samples == -1): 
        n_samples = len(fnames)
    n_test = int(n_samples*test_size)
    n_train = n_samples - n_test

    ## Split train and test samples
    # filenames
    fnames_train = fnames[0:n_train]
    fnames_test = fnames[n_train:n_train+n_test]
    # annotations
    annotations_train = annotations[0:n_train]
    annotations_test = annotations[n_train:n_train+n_test]
    # flags
    flags_train = bbox2d_exists[0:n_train]
    flags_test = bbox2d_exists[n_train:n_train+n_test]

    # return
    return fnames_train, annotations_train, flags_train, fnames_test, annotations_test, flags_test

# this function returns a generator for accessing data during training
def get_generator(fnames, annotations, flags, batch_size, dim=RESIZE_SHAPE):
    '''
    This function returns the generator for data and labels
    '''
    n_samples = len(fnames)
    while(1):
        # get a batch of data
        for offset in range(0, n_samples, batch_size):
            
            _X = []
            _Y = []

            img_fnames = fnames[offset:min(n_samples,offset+batch_size)]
            sub_annotations = annotations[offset:min(n_samples,offset+batch_size)]
            sub_flags = flags[offset:min(n_samples,offset+batch_size)]
            for i in range(len(img_fnames)):
                # make sure a corresponding depth file exists
                if(os.path.exists(img_fnames[i]) and (sub_flags[i] == True)):
                    # read, resize, and normalize image
                    img = cv2.cvtColor(cv2.imread(img_fnames[i]), cv2.COLOR_BGR2RGB)
                    img_size = [img.shape[0], img.shape[1]]
                    img = cv2.resize(img, dim)
                    img_norm = np.asarray(normalize_img(img), dtype=np.float32)
                    
                    # normalize bounding-boxes
                    label = assemble_label(sub_annotations[i], img_size=img_size)

                    _X.append(img_norm)
                    _Y.append(label) 
            
            yield np.asarray(_X), np.asarray(_Y)

# this function returns a train and a validation generator
def getDataGenerator(n_samples=-1, batch_size=16, test_size=0.1, data_path=data_path, dim=RESIZE_SHAPE, dataset='vkitti'):
    if dataset == 'vkitti':
        fnames_train, annotations_train, flags_train, fnames_test, annotations_test, flags_test = \
            assemble_data_vkitti(n_samples=n_samples, test_size=test_size, data_path=data_path)
    else:        
        fnames_train, annotations_train, flags_train, fnames_test, annotations_test, flags_test = \
            assemble_data_bbox_vector(n_samples=n_samples, test_size=test_size, data_path=data_path)

    # get train generator
    train_gen = get_generator(fnames_train, annotations_train, flags_train, batch_size=batch_size, dim=dim)
    # get test generator
    valid_gen = get_generator(fnames_test, annotations_test, flags_test, batch_size=batch_size, dim=dim)

    return train_gen, valid_gen

# this function returns data for training
def get_data(fnames, annotations, flags, dim=RESIZE_SHAPE):
    _X = []
    _Y = []

    for i in range(len(fnames)):
        # make sure a corresponding depth file exists
        if(os.path.exists(fnames[i]) and (flags[i] == True)):
            # read, resize, and normalize image
            img = cv2.cvtColor(cv2.imread(fnames[i]), cv2.COLOR_BGR2RGB)
            img_size = [img.shape[0], img.shape[1]]
            img = cv2.resize(img, dim)
            img_norm = np.asarray(normalize_img(img), dtype=np.float32)

            # normalize bounding-boxes
            label = assemble_label(annotations[i], img_size=img_size)

            _X.append(img_norm)
            _Y.append(label)
    
    return np.asarray(_X), np.asarray(_Y)

# this function computes and summarizes the data statistics
def get_data_statistics(annotations, out_fname='config/data_statistics.txt'):
    # list of parameters
    width, height, center_x, center_y = [], [], [], []

    # iterate through all the bounding-boxes
    for annotation in annotations:
        for bbox in annotation:
            width.append(float(bbox[2] - bbox[0]))
            height.append(float(bbox[3] - bbox[1]))
            center_x.append(float(bbox[0]) + (float(bbox[2] - bbox[0])/2.0))
            center_y.append(float(bbox[1]) + (float(bbox[3] - bbox[1])/2.0))
        
    # compute statistics
    center_x_mean = statistics.mean(center_x)
    center_x_stddev = statistics.stdev(center_x)
    center_y_mean = statistics.mean(center_y)
    center_y_stddev = statistics.stdev(center_y)
    width_mean = statistics.mean(width)
    width_stddev = statistics.stdev(width)
    height_mean = statistics.mean(height)
    height_stddev = statistics.stdev(height)

    # print statistics
    with open(out_fname, 'w') as f:
        print('---------------', file=f)
        print('Data Statistics', file=f)
        print('---------------', file=f)
        print('Center X - [Mean: {}],\t [Std Deviation: {}],\t [Min: {}],\t [MAX: {}]'.format(center_x_mean, center_x_stddev, min(center_x), max(center_x)), file=f)
        print('Center Y - [Mean: {}],\t [Std Deviation: {}],\t [Min: {}],\t [MAX: {}]'.format(center_y_mean, center_y_stddev, min(center_y), max(center_y)), file=f)
        print('Width    - [Mean: {}],\t [Std Deviation: {}],\t [Min: {}],\t [MAX: {}]'.format(width_mean, width_stddev, min(width), max(width)), file=f)
        print('Height   - [Mean: {}],\t [Std Deviation: {}],\t [Min: {}],\t [MAX: {}]'.format(height_mean, height_stddev, min(height), max(height)), file=f)

    os.system('cat '+out_fname)

# this function returns a train and validation data
def getData(n_samples=-1, test_size=0.1, data_path=data_path, dim=RESIZE_SHAPE, dataset='vkitti'):
    if dataset == 'vkitti':
        fnames_train, annotations_train, flags_train, fnames_test, annotations_test, flags_test = \
            assemble_data_vkitti(n_samples=n_samples, test_size=test_size, data_path=data_path)
    else:      
        fnames_train, annotations_train, flags_train, fnames_test, annotations_test, flags_test = \
            assemble_data_bbox_vector(n_samples=n_samples, test_size=test_size, data_path=data_path)
    # get train data
    x_train, y_train = get_data(fnames_train, annotations_train, flags_train, dim)
    # get test data
    x_test, y_test = get_data(fnames_test, annotations_test, flags_test, dim)

    return x_train, y_train, x_test, y_test

# this function returns number of train and test samples
def getNumSamples(n_samples=-1, test_size=0.1, data_path=data_path, dataset='vkitti'):
    if dataset == 'vkitti':
        _, _, flags_train, _, _, flags_test = \
            assemble_data_vkitti(n_samples=n_samples, test_size=test_size, data_path=data_path)
    else:        
        _, _, flags_train, _, _, flags_test = \
            assemble_data_bbox_vector(n_samples=n_samples, test_size=test_size, data_path=data_path)
    
    n_train_positive = np.count_nonzero(flags_train)
    n_test_positive = np.count_nonzero(flags_test)

    # return counts
    return n_train_positive, n_test_positive

if __name__ == "__main__":
    disp_sample = True
    dataset='vkitti'

    if dataset == 'vkitti':
        _, annotations, _, _, _, _ = \
            assemble_data_vkitti(n_samples=-1, test_size=0.0, data_path=data_path)
    else:
        _, annotations, _, _, _, _ = \
            assemble_data_bbox_vector(n_samples=-1, test_size=0.0, data_path=data_path)

    # get data statistics
    get_data_statistics(annotations)

    x_train, y_train, x_test, y_test = getData(n_samples=10, test_size=1.0, dataset=dataset)
    # print(y_test[0])

    # train_gen, valid_gen = getDataGenerator(n_samples=8, test_size=1.0)

    # x_train, y_train = next(train_gen)
    # x_test, y_test = next(train_gen)
    
    # plot one sample
    idx = 0
    # retrieve image
    img = denormalize_img(x_test[idx])
    # retrieve label
    label = y_test[idx]

    if(len(label) > 0):
        bboxes = disassemble_label(label, [img.shape[0], img.shape[1]])
        for grid_x in range(n_x_grids):
            for grid_y in range(n_y_grids):
                for n in range(n_anchors):
                    start_idx = (grid_x*n_y_grids+grid_y)*n_anchors*n_elements_per_anchor + (n*n_elements_per_anchor)
                    bbox = bboxes[start_idx:start_idx+n_elements_per_anchor]
                    conf = bbox[4]
                    if(conf > 0.5):
                        img = cv2.rectangle(img, (bbox[0],bbox[1]), (bbox[2],bbox[3]), color=(255,0,0), thickness=1)
        
        plt.imshow(img)
        plt.show()
