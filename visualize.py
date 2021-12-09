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
        square = 10
        print(b,d,h,w)
        ix = 1
        for i in range(square):
            for j in range(square):
                ax = plt.subplot(square, square, ix)
                f = features[0,:,:,ix-1]
                f = np.resize(f,(100,100))
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                plt.imshow(f, cmap='gray')
                ix += 1
        plt.savefig(os.path.join(self.visualize_dir,str(epoch)+".png"))
        print("save in : ",os.path.join(self.visualize_dir,str(epoch)+".png"))


from config.config import read_cfg
read_cfg()
from config.absl_mock import Mock_Flag
flag = Mock_Flag()
FLAGS = flag.FLAGS

if __name__ == '__main__':
    from helper_functions import *
    from byol_simclr_imagenet_data_harry import imagenet_dataset_single_machine
    import model_for_non_contrastive_framework as all_model




