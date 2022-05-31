from HARL.DataAugmentations.byol_simclr_imagenet_data_harry import imagenet_dataset_single_machine
import numpy as np
from absl import logging
import tensorflow as tf
import os
from imutils import paths
from absl import flags
import random
from sklearn.preprocessing import OneHotEncoder
import time
import glob
import os
import os
from config.absl_mock import Mock_Flag

from config.experiment_config import read_cfg
import matplotlib.pyplot as plt
read_cfg()
flag = Mock_Flag()
FLAGS = flag.FLAGS

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:

    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        print(e)




#strategy=None
strategy = tf.distribute.MirroredStrategy()
train_global_batch = 10
val_global_batch = 10
image_size = 224
bi_mask = False

train_global_batch = 10
val_global_batch = 10

# train_dataset = imagenet_dataset_single_machine(img_size=FLAGS.image_size, train_batch=train_global_batch,  val_batch=val_global_batch,
#                                                 strategy=strategy, train_path=FLAGS.train_path,
#                                                 val_path=FLAGS.val_path,
#                                                 mask_path=FLAGS.mask_path, bi_mask=False,
#                                                 train_label=FLAGS.train_label, val_label=FLAGS.val_label, subset_class_num=FLAGS.num_classes)

# train_ds = train_dataset.simclr_random_global_crop()


train_dataset = imagenet_dataset_single_machine(img_size=FLAGS.image_size, train_batch=train_global_batch,
                                                val_batch=val_global_batch,
                                                strategy=strategy, train_path=r"D:\OneDrive\鴻海\imagenet_1k_tiny\imagenet_1k_tiny\Image\train",
                                                val_path=None,
                                                mask_path=FLAGS.mask_path, bi_mask=True,
                                                train_label=FLAGS.train_label, val_label=FLAGS.val_label,
                                                subset_class_num=FLAGS.num_classes)


train_ds = train_dataset.simclr_inception_style_crop_image_mask()
# train_ds= train_dataset.simclr_random_global_crop_image_mask()

ds_1 = []
ds_2 =[]
for _, (ds_one, ds_two) in enumerate(train_ds):
    ds_1 = ds_one
    ds_2 = ds_two
    break

# image_mask, lable = ds
# image = image_mask[0]
# mask = image_mask[1]

# plt.figure(figsize=(10, 5))
# for n in range(10):
#     ax = plt.subplot(2, 10, n + 1)
#     plt.imshow(image[n])  # .numpy().astype("int")
#     ax = plt.subplot(2, 10, n + 11)
#     plt.imshow(tf.squeeze(mask[n])/255)  # .numpy().astype("int")
#     plt.axis("off")
# plt.show()
# print(image[0])

image,mask_,mask, lable = ds_1
image1,mask1_,mask1 ,_ = ds_2
# mask=image[1]
# image=image[0]
#
# mask1=image1[1]
# image1=image1[0]

plt.figure(figsize=(10, 5))
for n in range(8):
    ax = plt.subplot(2, 4, n + 1)
    if n<2: 
       plt.imshow(image[n])  # .numpy().astype("int")
        #print(mask.shape)
    elif 2<=n <4: 
        print(n-2)
        #print(image1.shape)
        plt.imshow(mask_[n-2])
    elif 4<= n <6: 
        print(n-4)
        #print(image1.shape)
        plt.imshow(image1[n-4])
    else: 
        print(n-6)
        #print(image1.shape)
        plt.imshow(mask1_[n-6])
    # ax = plt.subplot(2, 10, n + 11)
    # plt.imshow(tf.squeeze(mask[n])/255)  # .numpy().astype("int")
    plt.axis("off")

plt.show()
