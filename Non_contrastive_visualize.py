import os
import json
import math
import wandb
import random
# from absl import flags
from absl import logging
# from absl import app

import tensorflow as tf

from helper_functions import _restore_latest_or_from_pretrain
from learning_rate_optimizer import WarmUpAndCosineDecay
import metrics
from helper_functions import *
from byol_simclr_imagenet_data_harry import imagenet_dataset_single_machine
from self_supervised_losses import byol_symetrize_loss, symetrize_l2_loss_object_level_whole_image, sum_symetrize_l2_loss_object_backg, sum_symetrize_l2_loss_object_backg_add_original
import model_for_non_contrastive_framework as all_model
from visualize import Visualize

from config.config_56_56_512 import read_cfg
read_cfg()
from config.absl_mock import Mock_Flag
flag = Mock_Flag()
FLAGS = flag.FLAGS

import os
os.environ["CUDA_DEVICE_ORDER"]="0"
def main():

    strategy = tf.distribute.MirroredStrategy()
    train_global_batch = FLAGS.train_batch_size * strategy.num_replicas_in_sync
    val_global_batch = FLAGS.val_batch_size * strategy.num_replicas_in_sync

    train_dataset = imagenet_dataset_single_machine(img_size=FLAGS.image_size, train_batch=train_global_batch,  val_batch=val_global_batch,
                                                    strategy=strategy, train_path=r'D:\OneDrive\鴻海\SSL\Modify_code\imagenet_1k_tiny\imagenet_1k_tiny\Image\train',
                                                    val_path=None,
                                                    mask_path=FLAGS.mask_path, bi_mask=False,
                                                    train_label=FLAGS.train_label, val_label=FLAGS.val_label,
                                                    subset_class_num=FLAGS.num_classes)

    val_ds = train_dataset.supervised_validation()

    num_train_examples, num_eval_examples = train_dataset.get_data_size()
    from Model_resnet_harry import resnet
    model = resnet(resnet_depth=FLAGS.resnet_depth, width_multiplier=FLAGS.width_multiplier)
    model.build((1,224,224,3))
    from tensorflow.keras.models import load_model
    model.built = True
    weight_name = "56_56_512_binary"
    model.load_weights(os.path.join("D:/SSL_weight",weight_name,"encoder_model_99.h5"))
    model.summary()
    for i, (image, label) in enumerate(val_ds):
        #print("out put ",online_model.predict(image))
        # online_model.compile(optimizer='adam', loss='mse')
        import matplotlib.pyplot as plt
        print(image.shape)
        plt.imshow(image[0])
        plt.savefig(os.path.join(FLAGS.visualize_dir,"img" + ".png"))

        V = Visualize(1,FLAGS.visualize_dir)
        V.plot_feature_map(weight_name,model.predict(image))
        break

if __name__ == '__main__':
    main()
