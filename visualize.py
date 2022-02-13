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
#from helper_functions import *


class Visualize:
    def __init__(self, epoch, visualize_dir):
        self.epoch = epoch
        self.visualize_dir = visualize_dir

    def plot_feature_map(self, epoch, features, mask=None):
        square = 5
        ix = 1
        if mask != None:
            mask = tf.cast(mask, dtype=tf.bool)
            #mask = tf.logical_not(mask)
            mask = tf.cast(mask, dtype=features.dtype)
        for i in range(square):
            for j in range(square):
                ax = plt.subplot(square, square, ix)
                f = features[0, :, :, ix-1]
                f_max = np.max(f)
                f_min = np.min(f)
                f = (((f-f_min)/(f_max-f_min))*255.0).astype(np.uint8)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                if mask != None:
                    mask = tf.image.resize(mask, (f.shape[0], f.shape[1]))
                    f = tf.multiply(f, mask[:, :, 0])
                plt.imshow(f, cmap='gray')
                ix += 1
        plt.savefig(os.path.join(self.visualize_dir, str(epoch)+".png"))
        print("save in : ", os.path.join(self.visualize_dir, str(epoch)+".png"))
