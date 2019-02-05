import numpy as np
import os
from datetime import date
import math
import h5py
import random
import argparse


parser = argparse.ArgumentParser(description="Prepare the datasets")
parser.add_argument("in_dir", metavar="IN_DIR", type=str, help="Path to original images")

# data sets division settings
parser.add_argument("--regular_div", metavar="REGULAR_DIV", default=True, type=bool,
                    help="specify if the original image sequence is regularly separated.")
parser.add_argument("--day_block", metavar="DAY_BLOCK", default=7, type=str,
                    help="Specifies the time interval (how many days) within a train/val/test block.")
parser.add_argument("--ratio_div", metavar="RATIO_DIV", default=[0.6, 0.15, 0.25], type=tuple,
                    help="Specifies the ratio of splitting the train/val/test subsets within a block")
parser.add_argument("--order_div", metavar="ORDER_DIV", type=bool, default=True,
                    help="Specifies if the training, validation, and testing subsets are selected in order")
parser.add_argument("--end_date", metavar="END_DATE", type=list, default=[],
                    help="A list of file names that give the last date of each subset.")
parser.add_argument("--mode", metavar="MODE", type=str, default=None,
                    help="A list sequence of ['train','val','test'] that indicates the role of each specified subset")

# saving in the hdf5 file
parser.add_argument("--hdf5_dir", metavar="HDF5_DIR", default="../Data/", type=str, help="Path to the hdf5 file")
parser.add_argument("--hdf5_file", metavar="HDF5_NAME", default="", type=str, help="Name of the hdf5 file")


class DataSet:
    def __init__(self):
        self.train = []
        self.val = []
        self.test = []


class DataLoader(object):
    """
    Specifies training, validation, and testing sets
    Before running, check the file name formats!!!

    # Arguments
        in_dir: the directory of the data images.
        regular_div: specify if the original image sequence is regularly separated.
                     If TRUE, the image sequence is separated into regular time periods (train/val/test blocks)
                     If FALSE, the image sequence is separated into subsets with specified dates
                              (year_month_day, e.g. 2016_0101).

        # If TRUE is given for regular_div.
        day_block: Specifies the time interval (how many days) within a train/val/test block.
        ratio_div: Specifies the ratio of splitting the train/val/test subsets within a block,
                   A tuple of three float values add up to 1.
        order_div: Specifies if the training, validation, and testing subsets are selected in order (TRUE).
                   If FALSE, the subsets are specified randomly within a period.

        # If FALSE is given for regular_div.
        end_date: A list of file names that give the last date of each subset.
        mode: A list sequence of ['train','val','test'] that indicates the role of each specified subset.

    # Returns
        Lists for training/validation/testing sets.
    """

    def __init__(self, in_dir, regular_div=True, day_block=7, ratio_div=[0.6, 0.15, 0.25],
                 order_div=True, end_date=[], mode=None, remove_tr_list = []):
        self.in_dir = in_dir
        self.regular_div = regular_div
        self.day_block = day_block
        self.ratio_div = ratio_div
        self.order_div = order_div
        self.end_date = end_date
        self.remove_tr_list = remove_tr_list
        self.mode = mode
        self.t_file_list = self.__get_order()

    def generate(self):
        """Generates the .txt file of training, validation, and testing datasets"""

        if self.regular_div is True:
            data_set_list = self.__regular_generation(self.t_file_list)

        else:
            data_set_list = self.__irregular_generation(self.t_file_list)

        return data_set_list

    def save_hdf5(self, data_set_list, hdf5_file_dir, hdf5_file_name):
        # saving the list result in the file DataSet.hdf5
        # in data group "set", "index"
        hdf5_file = h5py.File(hdf5_file_dir + hdf5_file_name, 'r+')
        # data_set_list = self.generate()
        date0 = date(2016, 12, 1)
        for i in range(len(self.t_file_list[0])):
            im_name = hdf5_file["im_name"][i].decode("utf-8")
            im_time = (date(int(im_name[0:4]), int(im_name[5:7]), int(im_name[7:9]))-date0).days

            ind_train = [1 if im_name in i_period else 0 for i_period in data_set_list.train]
            if 1 in ind_train:
                im_period = ind_train.index(1)
                if im_name in self.remove_tr_list:
                    hdf5_file["set_index"][i, ...] = [3, im_period, data_set_list.train[im_period].index(im_name)]
                else:
                    hdf5_file["set_index"][i, ...] = [0, im_period, data_set_list.train[im_period].index(im_name)]
            else:
                ind_val = [1 if im_name in i_period else 0 for i_period in data_set_list.val]
                if 1 in ind_val:
                    im_period = ind_val.index(1)
                    # if im_time < 51:
                    hdf5_file["set_index"][i, ...] = [1, im_period, data_set_list.val[im_period].index(im_name)]
                    # else:
                    #     hdf5_file["set_index"][i, ...] = [3, im_period, data_set_list.val[im_period].index(im_name)]
                else:
                    ind_test = [1 if im_name in i_period else 0 for i_period in data_set_list.test]
                    if 1 in ind_test:
                        im_period = ind_test.index(1)
                        # if im_time < 51:
                        hdf5_file["set_index"][i, ...] = [2, im_period, data_set_list.test[im_period].index(im_name)]
                        # else:
                        #     hdf5_file["set_index"][i, ...] = [4, im_period,
                        #                                       data_set_list.test[im_period].index(im_name)]
                    # else:
                    #     hdf5_file["set_index"][i, ...] = [3,0,0]  # file not in data_set_list, should not happen

        hdf5_file.close()


    def __get_order(self):
        """
        Generates ordered list of files in in_dir, with corresponding time, according to time
        (unit: day, precise to hour. e.g. 12th day in the sequence, at 3:45, 12.15625)
        File name with format example: "2016_1230_08_30....png"
        """
        t_list = os.listdir(self.in_dir)
        t_list = [t_list[i][0:15] for i in range(len(t_list))]

        # date0 is a reference date for calculating the time difference,
        # tList1 contains the time difference of the files with date0
        date0 = date(2016, 12, 1)
        t_list1 = [(date(int(t_list[i][0:4]), int(t_list[i][5:7]), int(t_list[i][7:9]))-date0).days
                   + (float(t_list[i][10:12])+float(t_list[i][13:15])/60)/24 for i in range(len(t_list))]

        # sort the list according to time difference, subtract the time difference of the earliest file
        sort_ind = sorted(range(len(t_list1)), key=t_list1.__getitem__)
        t_file_list = [[t_list1[sort_ind[i]] - math.floor(t_list1[sort_ind[0]])
                        for i in range(len(sort_ind))], [t_list[sort_ind[i]] for i in range(len(sort_ind))]]

        return t_file_list   # [time, filename]

    def __regular_generation(self, t_file_list):
        data_set_list = DataSet()
        t_periods = []
        t_period = []
        i_period_day = t_file_list[0][0]  # the first day of the period
        # split the sequence in periods
        for i_file in range(len(t_file_list[0])):
            i_day = t_file_list[0][i_file]
            if i_day - i_period_day < self.day_block:
                t_period.append(t_file_list[1][i_file])
            else:
                t_periods.append(t_period)
                t_period = [t_file_list[1][i_file]]
                i_period_day = i_day
        t_periods.append(t_period)

        # split each period into training, validation, testing sets
        # in a order
        if self.order_div is True:
            for i_period in range(len(t_periods)):
                num_file = len(t_periods[i_period])
                num_train = int(round(num_file*self.ratio_div[0]))
                num_val = int(round(num_file*self.ratio_div[1]))

                data_set_list.train.append(t_periods[i_period][0:num_train])
                data_set_list.val.append(t_periods[i_period][num_train:num_train+num_val])
                data_set_list.test.append(t_periods[i_period][num_train+num_val:])
        # randomly
        else:
            for i_period in range(len(t_periods)):
                num_file = len(t_periods[i_period])
                ind_train = random.sample(range(0, num_file), int(round(num_file*self.ratio_div[0])))
                ind_val = random.sample(list(set(range(0, num_file))-set(ind_train)), int(round(num_file*self.ratio_div[1])))
                ind_test = list(set(range(0, num_file))-set(ind_val)-set(ind_train))
                ind_set = [ind_train, ind_val, ind_test]
                # ind_set = np.random.choice([0, 1, 2], num_file, p=self.ratio_div)
                i_sets = [[], [], []]

                for i_file in range(num_file):
                    i_file_set = [1 if i_file in i_set else 0 for i_set in ind_set]
                    i_file_set = i_file_set.index(1)
                    i_sets[i_file_set].append(t_periods[i_period][i_file])

                data_set_list.train.append(i_sets[0])
                data_set_list.val.append(i_sets[1])
                data_set_list.test.append(i_sets[2])

        return data_set_list

    def __irregular_generation(self, t_file_list):
        """
        Irregular division of data based on the end date of each subset given in self.end_date
        The subset contains all files to the last file with the end_date
        date format in string: year_month_day, e.g. 2016_0101
        """

        # Alert if the end_date is not given, or in the wrong format
        # Or the end dates are not given in order
        if len(self.end_date) == 0:
            print('Variable end_date not specified!')
            return

        if len(self.mode) != len(self.end_date):
            print('Variable end_date and mode should have the same length!')
            return

        data_set_list = DataSet()

        file0 = t_file_list[1][0]
        date0 = date(int(file0[0:4]),int(file0[5:7]),int(file0[7:9]))

        start_ind = 0  # the index of the first file of the subset
        for i_set in range(len(self.end_date)):
            i_end = self.end_date[i_set]
            i_day = (date(int(i_end[0:4]),int(i_end[5:7]),int(i_end[7:9])) - date0).days + 1
            # number of days after the first file date

            prev_list = i_day - np.asarray(t_file_list[0])
            end_ind = len(prev_list[prev_list>0])  # the index of the last file of the subset in the sequence

            i_mode = self.mode[i_set]
            if i_mode == 'train' or i_mode == 'training' or i_mode == 0:
                data_set_list.train.append(t_file_list[1][start_ind:end_ind])
            elif i_mode == 'val' or i_mode == 'validation' or i_mode == 1:
                data_set_list.val.append(t_file_list[1][start_ind:end_ind])
            elif i_mode == 'test' or i_mode == 'testing' or i_mode == 2:
                data_set_list.test.append(t_file_list[1][start_ind:end_ind])
            else:
                print('Items in variable mode not correct!')

            start_ind = end_ind

        return data_set_list


def main():
    args = parser.parse_args()
    # # ------------------------------------------------------------------------------------------------------------------ #
    # Generate training, validation, testing sets
    # the separation settings are meanwhile saved to the HDF5 file
    data_loader = DataLoader(args.in_dir, regular_div=args.regular_div, day_block=args.day_block,
                             ratio_div=args.ratio_div, order_div=args.order_div, end_date=args.end_date,
                             mode=args.mode)
    data_set_list = data_loader.generate()
    data_loader.save_hdf5(data_set_list, args.hdf5_dir, args.hdf5_file)


if __name__ == '__main__':
    main()
