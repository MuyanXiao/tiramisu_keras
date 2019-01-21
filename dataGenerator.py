import numpy as np
import math
import random
from data_loader import DataSet


class DataGenerator(object):
    """
    Generate batch sample on the fly, with the original data in an HDF5 file
    Given the data directory, the data_set_list
    Generate list of crop sets

    Generate augmented crop list
    """

    def __init__(self, hdf5_file, mean=None, dim_x=32, dim_y=32, dim_z=32, batch_size=32, shuffle=True, num_fusion=0, tag='Train', aug=0, balancing=True):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_fusion = num_fusion

        # with 56 patch size, the reduced image size would be different
        # so set to 784 and 1904
        self.r = hdf5_file["im"].shape[1]
        self.c = hdf5_file["im"].shape[2]
        self.aug = aug
        self.mean = mean
        self.balancing = balancing
        self.hdf5_file = hdf5_file
        self.tag = tag
        self.crop_list, self.num_crop_per_im, self.set_list = self.__get_crop_list()



    def generate(self):
        """
        __data_generation() to generate batch sample
        """
        # Infinite loop
        while True:
            # Generate order of exploration of data set
            indexes = self.__get_exploration_order()
            # Generate batches
            i_max = int(math.ceil(len(indexes)/float(self.batch_size)))

            for i in range(i_max):
                if i != (i_max -1):
                    # Find list of IDs
                    temp_ids_list = [self.crop_list[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]

                    # Generate data
                    X, y = self.__data_generation(temp_ids_list, self.mean, self.batch_size)
                    yield X, y

                else:
                    curr_batch_size = len(indexes)-i*self.batch_size
                    temp_ids_list = [self.crop_list[k] for k in indexes[i*self.batch_size:]]

                    # Generate data
                    X, y = self.__data_generation(temp_ids_list, self.mean, curr_batch_size)
                    yield X, y

    def __get_crop_list(self):
        """
        Given the directory, the set list, the patch dimension
        generate a total list of crops (augmented with random sampling and horizontal mirroring)

        crop_list
        [regularly cropped patches: im_name, r_start, c_start
        randomly cropped patches (random mirroring effect): im_name, r_start, c_start
        ...]

        hyper para setting: the number of augmented patches
        """

        crop_list = []
        overlap = self.dim_x/2
        num_crop_per_im = (int(math.floor(self.r/(self.dim_y - overlap)))-1)*(int(math.floor(self.c/(self.dim_x - overlap)))-1)
        im_name_list = self.hdf5_file["im_name"].value
        im_class_list = self.hdf5_file["im_class"].value
        set_index_list = self.hdf5_file["set_index"].value
        class_weight_0 = self.hdf5_file["class_weight"].value
        print(class_weight_0)

        # set_list: index of files in specific dataset
        # set_periods: e.g. number of images in each period in training set
        if self.tag == 'Train':
            self.set_ind_list = [i for i in range(len(set_index_list)) if set_index_list[i][0]==0]
        elif self.tag == 'Val':
            self.set_ind_list = [i for i in range(len(set_index_list)) if set_index_list[i][0]==1]
        elif self.tag == 'Test':
            self.set_ind_list = [i for i in range(len(set_index_list)) if set_index_list[i][0]==2]
        else:
            print('Wrong Tag!')

        set_name_order_list = sorted([im_name_list[i] for i in self.set_ind_list])

        # recover how the periods are split in the data_set_list
        nb_periods = 0
        set_periods = [0]
        for i in self.set_ind_list:
            i_period = set_index_list[i][1]
            if i_period > nb_periods:
                nb_periods = i_period

            if len(set_periods)-1 < i_period:
                for i in range(i_period + 1 - len(set_periods)):
                    set_periods.append(0)

            set_periods[i_period] += 1

        self.set_periods = set_periods

        set_list = recover_set_list(im_name_list, self.set_ind_list, set_index_list, self.set_periods)

        num_aug = get_num_aug(self.set_ind_list, im_class_list, self.aug, num_crop_per_im, balance=self.balancing)

        if self.num_fusion == 0:
            for ind_hdf5 in self.set_ind_list:
                if self.num_fusion != 0:
                    ind_in_period = set_index_list[ind_hdf5][2]
                    i_period = set_index_list[ind_hdf5][1]
                    if ind_in_period < self.num_fusion or ind_in_period > self.set_periods[i_period] - self.num_fusion - 1:
                        continue

                im_name = im_name_list[ind_hdf5]  # im_name has the format e.g. 2016_1201_08_30
                labels = self.hdf5_file["label"][ind_hdf5]
                # regular
                for i_crop in range(num_crop_per_im):
                    r_start, c_start = get_crop(self.dim_x, self.dim_y, self.c, overlap, i_crop)
                    if 0 not in labels[r_start:r_start+self.dim_y, c_start:c_start+self.dim_x]:
                        crop_list.append([im_name, r_start, c_start])
                    # crop_list.append([im_name, r_start, c_start])

                if self.aug != 0:
                    # random
                    # random sample the starting pixel position of the image

                    im_class = im_class_list[ind_hdf5]
                    crop_aug = random.sample(xrange(0, (self.r - self.dim_y) * (self.c - self.dim_x)), num_aug[im_class])
                    for i_crop_aug in crop_aug:
                        r_start = int(i_crop_aug/(self.c-self.dim_y))
                        c_start = int(i_crop_aug % (self.c-self.dim_y))
                        # if 0 not in labels[r_start:r_start + self.dim_y, c_start:c_start + self.dim_x]:
                        crop_list.append([im_name, r_start, c_start])

        else:
            for ind_hdf5 in self.set_ind_list:
                im_name = im_name_list[ind_hdf5]  # im_name has the format e.g. 2016_1201_08_30

                if set_name_order_list.index(im_name) > self.num_fusion:
                    im_0_name = set_name_order_list[set_name_order_list.index(im_name)-2*self.num_fusion]

                    if im_0_name[0:9] == im_name[0:9]:
                        labels = self.hdf5_file["label"][ind_hdf5]
                        # regular
                        for i_crop in range(num_crop_per_im):
                            r_start, c_start = get_crop(self.dim_x, self.dim_y, self.c, overlap, i_crop)
                            if 0 not in labels[r_start:r_start+self.dim_y, c_start:c_start+self.dim_x]:
                                crop_list.append([im_name, r_start, c_start])
                            # crop_list.append([im_name, r_start, c_start])

                        if self.aug != 0:
                            # random
                            # random sample the starting pixel position of the image

                            im_class = im_class_list[ind_hdf5]
                            crop_aug = random.sample(xrange(0, (self.r - self.dim_y) * (self.c - self.dim_x)), num_aug[im_class])
                            for i_crop_aug in crop_aug:
                                r_start = int(i_crop_aug/(self.c-self.dim_y))
                                c_start = int(i_crop_aug % (self.c-self.dim_y))
                                if 0 not in labels[r_start:r_start + self.dim_y, c_start:c_start + self.dim_x]:
                                    crop_list.append([im_name, r_start, c_start])

        print(len(crop_list))

        if self.mean is None:
            class_values = [127, 191, 255, 64]
            self.mean, self.class_weight = get_mean(self.dim_x, self.hdf5_file, crop_list, class_values)
            print(self.class_weight)

        return crop_list, num_crop_per_im, set_list

    def __get_exploration_order(self):
        # Generates order of exploration
        # Find exploration order
        indexes = np.arange(len(self.crop_list))
        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, temp_ids_list, mean, curr_batch_size):
        # Generates data of batch_size samples
        # X : (n_samples, v_size, v_size, v_size, n_channels)

        if self.num_fusion == 0:
            X = np.zeros((curr_batch_size, self.dim_x, self.dim_y, self.dim_z))
            # y = np.zeros((curr_batch_size, self.dim_x, self.dim_y))
        else:
            X = np.zeros((curr_batch_size, (2 * self.num_fusion + 1), self.dim_x, self.dim_y,  self.dim_z))
            # y = np.zeros((curr_batch_size, 2*self.num_fusion+1, self.dim_x, self.dim_y))

        y = np.zeros((curr_batch_size, self.dim_x, self.dim_y))

        # image name, set_index in HDF5
        im_name_list = self.hdf5_file["im_name"].value
        set_index_list = self.hdf5_file["set_index"].value

        # if_mirror = np.random.choice([0, 0], len(temp_ids_list), p=[0.5, 0.5])

        # Generate data
        for i, ID in enumerate(temp_ids_list):
            # get the image file position in the set list
            im_0_hdf5 = np.where(im_name_list == ID[0][0:15])[0][0]
            i_period = set_index_list[im_0_hdf5][1]
            i_im = set_index_list[im_0_hdf5][2]
            r_start = ID[1]
            c_start = ID[2]

            if self.num_fusion == 0:
                # index of the image in the HDF5 file
                patch = self.hdf5_file["im"][im_0_hdf5][r_start:r_start+self.dim_y, c_start:c_start+self.dim_x, :]-mean
                label = self.hdf5_file["label"][im_0_hdf5][r_start:r_start+self.dim_y, c_start:c_start+self.dim_x]
                X[i, :, :, :] = patch
                # if if_mirror[i] == 0:
                #     X[i, :, :, :] = patch
                #     y[i, :, :] = label
                # else:
                #     X[i, :, :, :] = np.fliplr(patch)
                #     y[i, :, :] = np.fliplr(label)

            else:
                for i_fuse in range(2 * self.num_fusion + 1):
                    im_i_hdf5 = np.where(im_name_list==self.set_list[i_period][i_im+i_fuse-2*self.num_fusion])[0][0]
                    patch = self.hdf5_file["im"][im_i_hdf5][r_start:r_start+self.dim_y, c_start:c_start+self.dim_x, :]
                    X[i, i_fuse, :, :, :] = patch - mean

                    # y[i, i_fuse, :, :] = self.hdf5_file["label"][im_i_hdf5][r_start:r_start+self.dim_y, c_start:c_start+self.dim_x]
            y[i, :, :] = self.hdf5_file["label"][im_0_hdf5][r_start:r_start+self.dim_y, c_start:c_start+self.dim_x]

        return X, sparse(y, self.num_fusion)


def sparse(y, num_fusion):
    # Returns labels in binary NumPy array
    color_classes = [127, 191, 255, 64]

    return np.array([[[[1 if y[i, r, c] == color_classes[j] else 0 for j in range(len(color_classes))]
                       for c in range(y.shape[2])] for r in range(y.shape[1])] for i in range(y.shape[0])])

    # if num_fusion == 0:
    #     return np.array([[[[1 if y[i, r, c] == color_classes[j] else 0 for j in range(len(color_classes))]
    #                        for c in range(y.shape[2])]for r in range(y.shape[1])]for i in range(y.shape[0])])
    # else:
    #     return np.array([[[[[1 if y[i, i_fuse, r, c]==color_classes[j] else 0 for j in range(len(color_classes))]
    #                         for c in range(y.shape[3])]for r in range(y.shape[2])]for i_fuse in range(y.shape[1])]for i in range(y.shape[0])])


def get_crop(dim_x, dim_y, c, overlap, id_crop):
    # calculate the start indexes of the crop
    c_crop = int(math.floor(c/(dim_x-overlap)))-1
    r_start = int(math.floor(id_crop/c_crop))*(dim_y-overlap)
    c_start = int(id_crop % c_crop)*(dim_x-overlap)

    return r_start, c_start


def get_mean(dim_patch, hdf5_file, crop_list, class_values=None):
    mean = np.zeros((dim_patch, dim_patch, 3))
    class_weight = np.zeros((4,))
    im_name_list = hdf5_file["im_name"].value

    for i in range(len(crop_list)):
        im_0_hdf5 = np.where(im_name_list == crop_list[i][0][0:15])[0][0]
        r_start = crop_list[i][1]
        c_start = crop_list[i][2]
        patch = hdf5_file["im"][im_0_hdf5][r_start:r_start+dim_patch, c_start:c_start+dim_patch, :]
        mean += patch

        label = hdf5_file["label"][im_0_hdf5][r_start:r_start+dim_patch, c_start:c_start+dim_patch]
        unique, counts = np.unique(label, return_counts=True)
        for unique_ind in range(len(unique)):
            class_ind = np.where(class_values == unique[unique_ind])
            class_ind = class_ind[0]
            class_weight[class_ind] += counts[unique_ind]

    mean = np.true_divide(mean, len(crop_list))
    class_weight = np.true_divide(sum(class_weight), class_weight)

    return mean, class_weight


def get_num_aug(set_ind_list, im_class_list, aug, crop_per_im, balance=True):
    if aug == 1 or balance == False:
        num_aug = [(aug-1)*crop_per_im, (aug-1)*crop_per_im, (aug-1)*crop_per_im]
    elif aug==0:
        num_aug = [0, 0, 0]
    else:
        num_aug = []
        set_class_list = np.array([im_class_list[i] for i in set_ind_list])
        im_class, nb_im_class = np.unique(set_class_list, return_counts=True)
        for i_class in range(3):  # water, ice, snow
            nb_class = nb_im_class[np.where(im_class==i_class)]
            num_aug_class = (aug*len(set_ind_list)/3 - nb_class) * (aug-1) * crop_per_im
            num_aug_class = num_aug_class/nb_class
            num_aug.append(num_aug_class[0])

    print(num_aug)
    return num_aug


def recover_set_list(im_name_list, set_ind_list, set_index_list, set_periods):
    set_list = [['_' for i in range(i_period)] for i_period in set_periods]
    for id_file in set_ind_list:
        i_period = set_index_list[id_file][1]
        id_in_period = set_index_list[id_file][2]
        set_list[i_period][id_in_period] = im_name_list[id_file]

    return set_list
