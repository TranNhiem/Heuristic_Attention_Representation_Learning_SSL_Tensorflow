import numpy as np
import tensorflow as tf
from tensorflow import keras

# Display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import os

import math
import errno
import shutil
from helper_functions import *

class Visualize:
    def __init__(self,epoch,visualize_dir):
        self.epoch = epoch
        self.visualize_dir = visualize_dir

    def plot_feature_map(self,epoch,features):
        b,h,w,d= features.shape
        print(b,d,h,w)
        for i in range(b):
            fig, ax = plt.subplots(1,d)
            for j in range(d):
                f = features[i,:,:,j]
                print(type(f))
                proto_tensor = tf.make_tensor_proto(f)
                f = tf.make_ndarray(proto_tensor)
                ax[j].imshow(f)
            plt.savefig(os.path.join(self.visualize_dir,str(epoch)+"_"+str(i)+".png"))
            print("save in : ",os.path.join(self.visualize_dir,str(epoch)+"_"+str(i)+".png"))


    # def get_grid_dim(self,x):
    #     """
    #     Transforms x into product of two integers
    #     :param x: int
    #     :return: two ints
    #     """
    #     factors = self.prime_powers(x)
    #     if len(factors) % 2 == 0:
    #         i = int(len(factors) / 2)
    #         return factors[i], factors[i - 1]

    #     i = len(factors) // 2
    #     return factors[i], factors[i]


    # def prime_powers(self,n):
    #     """
    #     Compute the factors of a positive integer
    #     Algorithm from https://rosettacode.org/wiki/Factors_of_an_integer#Python
    #     :param n: int
    #     :return: set
    #     """
    #     factors = set()
    #     for x in range(1, int(math.sqrt(n)) + 1):
    #         if n % x == 0:
    #             factors.add(int(x))
    #             factors.add(int(n // x))
    #     return sorted(factors)


    # def empty_dir(self,path):
    #     """
    #     Delete all files and folders in a directory
    #     :param path: string, path to directory
    #     :return: nothing
    #     """
    #     for the_file in os.listdir(path):
    #         file_path = os.path.join(path, the_file)
    #         try:
    #             if os.path.isfile(file_path):
    #                 os.unlink(file_path)
    #             elif os.path.isdir(file_path):
    #                 shutil.rmtree(file_path)
    #         except Exception as e:
    #             print('Warning: {}'.format(e))


    # def create_dir(self,path):
    #     """
    #     Creates a directory
    #     :param path: string
    #     :return: nothing
    #     """
    #     try:
    #         os.makedirs(path)
    #     except OSError as exc:
    #         if exc.errno != errno.EEXIST:
    #             raise


    # def prepare_dir(self,path, empty=False):
    #     """
    #     Creates a directory if it soes not exist
    #     :param path: string, path to desired directory
    #     :param empty: boolean, delete all directory content if it exists
    #     :return: nothing
    #     """
    #     if not os.path.exists(path):
    #         self.create_dir(path)

    #     if empty:
    #         self.empty_dir(path)

    # def plot_feature_map(self, name,conv_img):
        # """
        # Makes plots of results of performing convolution
        # :param conv_img: numpy array of rank 4
        # :param name: string, name of convolutional layer
        # :return: nothing, plots are saved on the disk
        # """
        # # make path to output folder
        # plot_dir = os.path.join(self.visualize_dir, 'conv_output')
        # plot_dir = os.path.join(plot_dir, name)

        # # create directory if does not exist, otherwise empty it
        # self.prepare_dir(plot_dir, empty=True)

        # w_min = np.min(conv_img)
        # w_max = np.max(conv_img)

        # # get number of convolutional filters
        # num_filters = conv_img.shape[3]

        # # get number of grid rows and columns
        # grid_r, grid_c = self.get_grid_dim(num_filters)

        # # create figure and axes
        # fig, axes = plt.subplots(min([grid_r, grid_c]),
        #                         max([grid_r, grid_c]))

        # # iterate filters
        # for l, ax in enumerate(axes.flat):
        #     # get a single image
        #     img = conv_img[0, :, :,  l]
        #     # put it on the grid
        #     ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='bicubic', cmap='Greys')
        #     # remove any labels from the axes
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        # # save figure
        # plt.savefig(os.path.join(plot_dir, '{}.png'.format(name)), bbox_inches='tight')

