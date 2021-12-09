import os
import json
import math
import wandb
import random
# from absl import flags
from absl import logging
# from absl import app

import tensorflow as tf
from learning_rate_optimizer import WarmUpAndCosineDecay
import metrics
from helper_functions import *
from byol_simclr_imagenet_data_harry import imagenet_dataset_single_machine
from self_supervised_losses import byol_symetrize_loss, symetrize_l2_loss_object_level_whole_image, sum_symetrize_l2_loss_object_backg, sum_symetrize_l2_loss_object_backg_add_original
import model_for_non_contrastive_framework as all_model
from visualize import Visualize

from config.config_visualize import read_cfg
read_cfg()
from config.absl_mock import Mock_Flag
flag = Mock_Flag()
FLAGS = flag.FLAGS


def main():

    strategy = tf.distribute.MirroredStrategy()
    train_global_batch = FLAGS.train_batch_size * strategy.num_replicas_in_sync
    val_global_batch = FLAGS.val_batch_size * strategy.num_replicas_in_sync

    train_dataset = imagenet_dataset_single_machine(img_size=FLAGS.image_size, train_batch=train_global_batch,  val_batch=val_global_batch,
                                                    strategy=strategy, train_path=FLAGS.train_path,
                                                    val_path=FLAGS.val_path,
                                                    mask_path=FLAGS.mask_path, bi_mask=True,
                                                    train_label=FLAGS.train_label, val_label=FLAGS.val_label,
                                                    subset_class_num=FLAGS.num_classes)

    train_ds = train_dataset.simclr_inception_style_crop_image_mask()

    val_ds = train_dataset.supervised_validation()

    num_train_examples, num_eval_examples = train_dataset.get_data_size()

    train_steps = FLAGS.eval_steps or int(
        num_train_examples * FLAGS.train_epochs // train_global_batch)*2
      
    eval_steps = FLAGS.eval_steps or int(
        math.ceil(num_eval_examples / val_global_batch))

    epoch_steps = int(round(num_train_examples / train_global_batch))

    checkpoint_steps = (FLAGS.checkpoint_steps or (
        FLAGS.checkpoint_epochs * epoch_steps))

    # Configure the Encoder Architecture.
    online_model = all_model.Binary_online_model(FLAGS.num_classes,Upsample = FLAGS.feature_upsample,Downsample = FLAGS.downsample_mod)
    prediction_model = all_model.prediction_head_model()
    target_model = all_model.Binary_target_model(FLAGS.num_classes,Upsample = FLAGS.feature_upsample,Downsample = FLAGS.downsample_mod)

    # Configure the learning rate
    base_lr = FLAGS.base_lr
    scale_lr = FLAGS.lr_rate_scaling
    warmup_epochs = FLAGS.warmup_epochs
    train_epochs = FLAGS.train_epochs

    lr_schedule = WarmUpAndCosineDecay(
        base_lr, train_global_batch, num_train_examples, scale_lr, warmup_epochs,
        train_epochs=train_epochs, train_steps=train_steps)

    # Current Implement the Mixpercision optimizer
    optimizer = all_model.build_optimizer(lr_schedule)

    # Build tracking metrics
    all_metrics = []
    # Linear classfiy metric
    weight_decay_metric = tf.keras.metrics.Mean('train/weight_decay')
    total_loss_metric = tf.keras.metrics.Mean('train/total_loss')
    all_metrics.extend([weight_decay_metric, total_loss_metric])

    if FLAGS.train_mode == 'pretrain':
        # for contrastive metrics
        contrast_loss_metric = tf.keras.metrics.Mean(
            'train/non_contrast_loss')
        contrast_acc_metric = tf.keras.metrics.Mean(
            "train/non_contrast_acc")
        contrast_entropy_metric = tf.keras.metrics.Mean(
            'train/non_contrast_entropy')
        all_metrics.extend(
            [contrast_loss_metric, contrast_acc_metric, contrast_entropy_metric])

    if FLAGS.train_mode == 'finetune' or FLAGS.lineareval_while_pretraining:
        logging.info(
            "Apllying pre-training and Linear evaluation at the same time")
        # Fine-tune architecture metrics
        supervised_loss_metric = tf.keras.metrics.Mean(
            'train/supervised_loss')
        supervised_acc_metric = tf.keras.metrics.Mean(
            'train/supervised_acc')
        all_metrics.extend(
            [supervised_loss_metric, supervised_acc_metric])

    # Check and restore Ckpt if it available
    # Restore checkpoint if available.
    checkpoint_manager = try_restore_from_checkpoint(
        online_model, optimizer.iterations, optimizer)
    print(online_model.layers[0].name)
    for i, (image, label) in enumerate(val_ds):
        #print("out put ",online_model.predict(image))
        online_model.compile(optimizer='adam', loss='mse')
        import keract
        # activations = keract.get_activations(online_model, image)
        # keract.display_heatmaps(activations, image, save=True)

        V = Visualize(1,FLAGS.visualize_dir)
        V.plot_feature_map("28_28_1024",online_model.predict(image))
        break
            # out = image
            # for i,layer in enumerate(online_model.layers):
            #     out = online_model.get_layer(layer.name)(out)
            #     print(image.shape)
            #     print(label.shape)
            #     if i == 0:
            #         print(out)
            #         break


        #
        # online_model.layers
        # for l in online_model.layers:
        #     print(l)
        #



if __name__ == '__main__':
    main()
