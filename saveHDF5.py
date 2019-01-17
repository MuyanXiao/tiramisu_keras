import numpy as np
import cv2
import math
import h5py
import os


class SaveHDF5(object):
    """
    Save the image and labels in hdf5
    The file has 7 data groups:

    "im": image data
    "label": label image data
    "im_name": name of image file (e.g. 2016_1204_08_30)
    "set_index": indicates which set the image file belongs to
                 (0, 1, 2 represents training, validation, testing respectively)
                 tuple of integers giving the index of the image file in the set construction ([set, i_period, i_file])
    "class_weighting": the weight of the classes, calculated from the training set
    "mean": mean patch value of each data set

    "im", "label", "im_name" are defined in writeHDF5, and stays the same with the actual data
    "set", "index", "class_weighting", "mean" changes with the data_loader setting
    "set" and "index" are defined in data_loader
    "class_weighting", "mean" are calculated in dataGenerator
    """

    def __init__(self, in_dir, label_dir, out_dir, out_name):
        im_list = os.listdir(label_dir)
        im0 = cv2.imread(in_dir+im_list[0])  # read an image to get the image dimension
        # (self.r, self.c, _) = im0.shape
        self.r = 324
        self.c = 1209
        num_im = len(im_list)

        # HDF5 file settings
        hdf5_path = out_dir+out_name
        self.hdf5_file = h5py.File(hdf5_path, mode='w')

        self.hdf5_file.create_dataset("im", (num_im, self.r, self.c, 3), np.uint8)
        self.hdf5_file.create_dataset("label", (num_im, self.r, self.c), np.uint8)
        self.hdf5_file.create_dataset("im_name", (num_im,), "S15")
        self.hdf5_file.create_dataset("set_index", (num_im, 3), np.int16)
        self.hdf5_file.create_dataset("im_class", (num_im,), np.int)
        self.hdf5_file.create_dataset("class_weight", (4,), np.float16)

        self.in_dir = in_dir
        self.im_list = im_list
        self.label_dir = label_dir

    def writeHDF5(self):
        """
        write data in the hdf5 file (data group "im", "label", "im_name")
        """
        class_values = [127, 191, 255, 64]

        for i in range(len(self.im_list)):
            im = cv2.imread(self.in_dir + self.im_list[i])[0:self.r, 0:self.c, :]
            im_label = cv2.imread(self.label_dir + self.im_list[i])[0:self.r, 0:self.c, 0]

            self.hdf5_file["im"][i, ...] = im[None]
            self.hdf5_file["label"][i, ...] = im_label[None]

            unique, counts = np.unique(im_label, return_counts=True)
            self.hdf5_file["im_class"][i] = np.where(class_values == unique[np.argmax(counts)])[0][0]

            self.hdf5_file["im_name"][i] = self.im_list[i][0:15]

        self.hdf5_file.close()



