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
                #Sf = f.eval(session=tf.compat.v1.Session())
                f = f.numpy()
                print(type(f))
            # proto_tensor = tf.make_tensor_proto(f)
            # f = tf.make_ndarray(proto_tensor)
                ax[j].imshow(f)
            plt.savefig(os.path.join(self.visualize_dir,str(epoch)+"_"+str(i)+".png"))
            print("save in : ",os.path.join(self.visualize_dir,str(epoch)+"_"+str(i)+".png"))


from config.config import read_cfg
read_cfg()
from config.absl_mock import Mock_Flag
flag = Mock_Flag()
FLAGS = flag.FLAGS

if __name__ == '__main__':
    print()
    from helper_functions import *
    from byol_simclr_imagenet_data_harry import imagenet_dataset_single_machine
    import model_for_non_contrastive_framework as all_model




