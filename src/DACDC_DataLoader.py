import os
import cv2
import matplotlib.pyplot as plt
from src.dacdc_utils import *

class DACDC_DataLoader():
    def __init__(self, data_path, labels_fname):
        self.data_path = data_path
        self.labels_fname = self.data_path + labels_fname

    # normalize image
    def normalize_img(self, X):
        return (np.true_divide(X, 255.0) - 0.5)

    # denormalize image
    def denormalize_img(self, X):
        return np.asarray(np.multiply((X + 0.5), 255.0), dtype=np.uint8)

    def assemble_annotations(self, n_samples=-1, test_size=0.1, shuffle=True):
        lines = []
        # Open annotated 2D bounding-boxes
        with open(self.labels_fname, 'r') as file:
            lines = file.readlines()

        fnames, classes_list, bboxes_list = [], [], []
        for i in range(len(lines)):
            classes, bboxes = [], []
            line = lines[i].replace('\n', '').split()
            n_objs = int(line[1])
            if(n_objs > 0):
                fnames.append(self.data_path+str(line[0]))
                # get a list of 2d bounding-boxes
                for j in range(n_objs):
                    classes.append( str(line[2 + 5*j]) )
                    bboxes.append([ float(line[3 + 5*j]),
                                    float(line[4 + 5*j]),
                                    float(line[5 + 5*j]),
                                    float(line[6 + 5*j])])

            assert(len(classes) == len(bboxes))
            classes_list.append(classes)
            bboxes_list.append(bboxes)

        # shuffle the data
        assert len(fnames) == len(classes_list)
        assert len(fnames) == len(bboxes_list)
        p = np.random.permutation(len(fnames))
        fnames = np.asarray(fnames)[p]
        classes_list = np.asarray(classes_list)[p]
        bboxes_list = np.asarray(bboxes_list)[p]

        # compute number of train and test samples
        if(n_samples >= len(fnames)) or (n_samples == -1): 
            n_samples = len(fnames)

        n_test = int(float(n_samples)*test_size)
        n_train = n_samples - n_test
            
        ## Split train and validation samples
        # filenames
        fnames_train = fnames[0:n_train]
        fnames_test = fnames[n_train:n_train+n_test]
        # classes
        classes_train = classes_list[0:n_train]
        classes_test = classes_list[n_train:n_train+n_test]
        # bounding-boxes
        bboxes_train = bboxes_list[0:n_train]
        bboxes_test = bboxes_list[n_train:n_train+n_test]

        return fnames_train, classes_train, bboxes_train, fnames_test, classes_test, bboxes_test

    def _get_data(self, fnames, classes, bboxes, dim=RESIZE_SHAPE):
        _X = []
        _Y = []

        for i in range(len(fnames)):
            # make sure a corresponding file exists
            if(os.path.exists(fnames[i])):
                # read, resize, and normalize image
                img = cv2.cvtColor(cv2.imread(fnames[i]), cv2.COLOR_BGR2RGB)
                img_size = [img.shape[0], img.shape[1]]
                img = cv2.resize(img, dim)
                img_norm = np.asarray(self.normalize_img(img), dtype=np.float32)

                # normalize bounding-boxes
                label = assemble_label(classes[i], bboxes[i], img_size=img_size)

                _X.append(img_norm)
                _Y.append(label)
        
        return np.asarray(_X), np.asarray(_Y)

    def get_data(self, dim=RESIZE_SHAPE, n_samples=-1, test_size=0.1, shuffle=True):
        fnames_train, classes_train, bboxes_train, fnames_test, classes_test, bboxes_test = \
            self.assemble_annotations(n_samples=n_samples, test_size=test_size, shuffle=shuffle)

        x_train, y_train = self._get_data(fnames_train, classes_train, bboxes_train, dim=dim)
        x_test, y_test = self._get_data(fnames_test, classes_test, bboxes_test, dim=dim)

        return x_train, y_train, x_test, y_test

    # this function returns a generator for accessing data 
    def _get_generator(self, fnames, classes, bboxes, batch_size, dim=RESIZE_SHAPE):
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
                sub_classes = classes[offset:min(n_samples,offset+batch_size)]
                sub_bboxes = bboxes[offset:min(n_samples,offset+batch_size)]
                for i in range(len(img_fnames)):
                    # make sure a corresponding depth file exists
                    if(os.path.exists(img_fnames[i])):
                        # read, resize, and normalize image
                        img = cv2.cvtColor(cv2.imread(img_fnames[i]), cv2.COLOR_BGR2RGB)
                        img_size = [img.shape[0], img.shape[1]]
                        img = cv2.resize(img, dim)
                        img_norm = np.asarray(self.normalize_img(img), dtype=np.float32)
                        
                        # normalize bounding-boxes
                        label = assemble_label(sub_classes[i], sub_bboxes[i], img_size=img_size)

                        _X.append(img_norm)
                        _Y.append(label)
                
                yield np.asarray(_X), np.asarray(_Y)

    def get_data_generator(self, n_samples=-1, batch_size=16, test_size=0.1, dim=RESIZE_SHAPE, shuffle=True):
        fnames_train, classes_train, bboxes_train, fnames_test, classes_test, bboxes_test = \
            self.assemble_annotations(n_samples=n_samples, test_size=test_size, shuffle=shuffle)

        # get train generator
        train_gen = self._get_generator(fnames_train, classes_train, bboxes_train, batch_size=batch_size, dim=dim)

        # get test generator
        test_gen = self._get_generator(fnames_test, classes_test, bboxes_test, batch_size=batch_size, dim=dim)

        return train_gen, test_gen

    def get_num_samples(self, n_samples=-1, test_size=0.1):
        fnames_train, _, _, fnames_test, _, _ = \
            self.assemble_annotations(n_samples=n_samples, test_size=test_size)

        n_train_positive = len(fnames_train)
        n_test_positive = len(fnames_test)

        # return counts
        return n_train_positive, n_test_positive


if __name__ == '__main__':
    dacdc_dloader = DACDC_DataLoader(   '/home/shubham/workspace/dataset/KITTI/data_object_image_2/training/',
                                        'dacdc_labels.txt')
    # x_train, y_train, x_test, y_test = dacdc_dloader.get_data(n_samples=10)
    train_gen, test_gen = dacdc_dloader.get_data_generator(n_samples=10)
    x_train, y_train = next(train_gen)
    x_test, y_test = next(test_gen)

    # display one sample
    test_idx = 0
    test_img = dacdc_dloader.denormalize_img(x_test[test_idx])
    conf, bboxes, classes = labels2bboxes(y_test[test_idx], img_size=RESIZE_SHAPE)

    for k, bbox in enumerate(bboxes):
        # draw bbox if conf is > 50%
        if(conf[k] > 0.5):
            test_img = cv2.rectangle(test_img, (bbox[0],bbox[1]), (bbox[2],bbox[3]), color=(255,0,0), thickness=2)
            test_img = cv2.rectangle(test_img, (bbox[0],bbox[1]-15), (bbox[0]+80,bbox[1]-2), color=(0,0,0), thickness=-1)
            if CLASS_PREDICTION == True:
                cv2.putText(test_img,classes[k]+' {:.1f}%'.format(conf[k][0]*100.0), 
                (bbox[0], bbox[1]-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.3,
                (255,255,255),
                1,
                2)
            else:
                cv2.putText(test_img,'Conf {:.1f}%'.format(conf[k][0]*100.0), 
                (bbox[0], bbox[1]-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.3,
                (255,255,255),
                1,
                2)
    
    plt.imshow(test_img)
    plt.show()
