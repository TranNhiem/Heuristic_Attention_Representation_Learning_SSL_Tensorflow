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

import matplotlib.pyplot as plt


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:

    try:
        tf.config.experimental.set_visible_devices(gpus[0:2], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        print(e)


# flags.DEFINE_integer(
#     'SEED_data_split', 100,
#     'random seed for spliting data.')

imagenet_path = "/data/SSL_dataset/ImageNet/1K/"
# binary_mask= "/data/SSL_dataset/ImageNet/binary_image_by_USS/"


dataset = list(paths.list_images(imagenet_path))
# dataset_mask= list(paths.list_images(binary_mask))


# dataset=[]
# dataset_mask=[]
# for name in glob(os.listdir(imagenet_path)):
#     dataset.append(name)
# for name in glob(os.listdir(binary_mask)):
#     dataset_mask.append(name)
random.Random(100).shuffle(dataset)
dataset_mask=[]
for path in dataset: 
    dataset_mask.append(path.replace("1K/", "binary_image_by_USS/").replace("JPEG","png"))

x_val = dataset[0:50000]
x_train = dataset[50000:200000]
# x_binary = dataset_mask[50000:200000]
# print(x_train[1:20])
# print(x_binary[1:20])
# x_image_mask= zip(x_train, x_binary)
# i=0
# for path in x_image_mask: 
#     image, mask= path[0], path[1]

#     print(image)
#     print(mask)
#     i+=1
#     if i==10: 
#         break



# # Encode all Class
# all_train_class = []
# for image_path in x_train:
#     #label = tf.strings.split(image_path, os.path.sep)[5]
#     label= image_path.split("/")[5]
#     all_train_class.append(label)

# print(all_train_class[1:100])
# number_class = set(all_train_class)
# all_cls = list(number_class)

# all_val_class = []
# for image_path in x_val:
#     #label = tf.strings.split(image_path, os.path.sep)[5]
#     label= image_path.split("/")[5]
#     all_val_class.append(label)

# number_val_class = set(all_val_class)
# all_val_cls = list(number_val_class)
# # logging.info("number class in training data", all_cls)
# # logging.info("number class in validation data", all_val_cls)
# print("number class in training data", len(all_cls))
# print("number class in validation data",len(all_val_cls))
# # Mapping class to It ID
# class_dic = dict()
# for i in range(999):
#     class_dic[all_cls[i]] = i+1

# numeric_train_cls = []
# for i in range(len(all_train_class)):
#     for k, v in class_dic.items():
#         if all_train_class[i] == k:
#             numeric_train_cls.append(v)

# start_time= time.time()
# one_hot_encode_cls_tf = tf.one_hot(numeric_train_cls, depth=999)
# print("Time complete onehot_Ecode tensorflow", time.time() - start_time)
# # start_time_1= time.time()
# # enc = OneHotEncoder(handle_unknown='ignore')
# # one_hot_encode_cls= enc.fit_transform([numeric_train_cls])
# # print("Time complete onehot_Ecode sklearn", time.time() - start_time_1)
# # one_hot_encode_cls

# print(one_hot_encode_cls[1000:1050])
# # logging.info("number of training sample", len(one_hot_encode_cls))
# # logging.info("number of validation sample", len(all_val_class))
# print("number of training sample", len(one_hot_encode_cls))
# print("number of validation sample", len(all_val_class))


import numpy as np
from byol_simclr_imagenet_data import imagenet_dataset_single_machine
strategy = tf.distribute.MirroredStrategy()
train_global_batch=32
val_global_batch=32
image_size=224
bi_mask=True
train_dataset = imagenet_dataset_single_machine(img_size=image_size, train_batch=train_global_batch,  val_batch=val_global_batch,
                                                strategy=strategy, img_path=None, x_val=x_val,  x_train=x_train, bi_mask=True)


train_ds = train_dataset.simclr_random_global_crop_image_mask()

ds=[]
for _, (ds_one, ds_two) in enumerate(train_ds):
    ds= ds_one
    
    break
    
#val_ds = train_dataset.supervised_validation()

# for _, (ds_one, ds_two) in enumerate(train_ds):
#     # ds_one=np.array(ds_one)
#     # print(ds_one.shape) 

image_mask, lable = ds
image=image_mask[0]
mask=image_mask[1]
#print(label)

plt.figure(figsize=(10, 5))
for n in range(10):
    ax = plt.subplot(2, 10, n + 1)
    plt.imshow(image[n])#.numpy().astype("int")
    ax = plt.subplot(2, 10, n + 11)
    plt.imshow(tf.squeeze(mask[n])/255)#.numpy().astype("int")
    plt.axis("off")
plt.show()
# break
# image, mask, label= next(iter(ds_one))
# print(image.shape)
# print(mask.shape)
# print(label.shape)