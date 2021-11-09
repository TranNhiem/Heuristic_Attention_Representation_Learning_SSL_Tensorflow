from absl import logging
import tensorflow as tf
import os
from imutils import paths
from absl import flags
import random


FLAGS = flags.FLAGS


# flags.DEFINE_integer(
#     'SEED_data_split', 100,
#     'random seed for spliting data.')


imagenet_path = "/data/SSL_dataset/ImageNet/1K/"
dataset = list(paths.list_images(imagenet_path))
random.Random(100).shuffle(dataset)
x_val = dataset[0:50000]
x_train = dataset[50000:200000]
print(x_train[1:10])


# Encode all Class
all_train_class = []
for image_path in x_train:
    label = tf.strings.split(image_path, os.path.sep)[4]
    all_train_class.append(label.numpy())
print(all_train_class[1:100])
number_class = set(all_train_class)
all_cls = list(number_class)

all_val_class = []
for image_path in x_val:
    label = tf.strings.split(image_path, os.path.sep)[4]
    all_val_class.append(label.numpy())

number_val_class = set(all_val_class)
all_val_cls = list(number_val_class)
logging.info("number class in training data", all_cls)
logging.info("number class in validation data", all_val_cls)
# Mapping class to It ID
class_dic = dict()
for i in range(1000):
    class_dic[all_cls[i]] = i+1

numeric_train_cls = []
for i in range(len(all_train_class)):
    for k, v in class_dic.items():
        if all_train_class[i] == k:
            numeric_train_cls.append(v)

one_hot_encode_cls = tf.one_hot(numeric_train_cls, depth=1000)
one_hot_encode_cls
print(one_hot_encode_cls[1000:1050])
logging.info("number of training sample", len(one_hot_encode_cls))
logging.info("number of validation sample", len(all_val_class))
