from config.absl_mock import Mock_Flag
from config.experiment_config_multi_machine import read_cfg
import os
import json
import math
import wandb
import random
from absl import flags
from absl import logging
from absl import app

import tensorflow as tf
from learning_rate_optimizer import WarmUpAndCosineDecay
import metrics
from helper_functions import *
from multi_machine_dataloader import imagenet_dataset_multi_machine
from self_supervised_losses import byol_symetrize_loss, symetrize_l2_loss_object_level_whole_image, \
    sum_symetrize_l2_loss_object_backg, sum_symetrize_l2_loss_object_backg_add_original, byol_loss
import model_for_non_contrastive_framework as all_model
import objective as obj_lib
from imutils import paths

# from tensorflow.keras import mixed_precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.pop('TF_CONFIG', None)

tf.keras.backend.clear_session()
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


read_cfg()
flag = Mock_Flag()
FLAGS = flag.FLAGS

if not os.path.isdir(FLAGS.model_dir):
    print("creat : ", FLAGS.model_dir, FLAGS.cached_file_val, FLAGS.cached_file)
    os.makedirs(FLAGS.model_dir)


flag.save_config(os.path.join(FLAGS.model_dir, "config.cfg"))

# For setting GPUs Thread reduce kernel Luanch Delay
# https://github.com/tensorflow/tensorflow/issues/25724
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_GPU_THREAD_COUNT'] = '2'


def main():
    # ------------------------------------------
    # Communication methods
    # ------------------------------------------
    if FLAGS.communication_method == "NCCL":

        communication_options = tf.distribute.experimental.CommunicationOptions(
            implementation=tf.distribute.experimental.CommunicationImplementation.NCCL)

    elif FLAGS.communication_method == "RING":

        communication_options = tf.distribute.experimental.CommunicationOptions(
            implementation=tf.distribute.experimental.CommunicationImplementation.RING)

    elif FLAGS.communication_method == "auto":
        communication_options = tf.distribute.experimental.CommunicationOptions(
            implementation=tf.distribute.experimental.CollectiveCommunication.AUTO)

    else:
        raise ValueError("Invalida communication method")
    # strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
    # communication=tf.distribute.experimental.CollectiveCommunication.AUTO,
    # cluster_resolver=None)
    resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
    strategy = tf.distribute.MultiWorkerMirroredStrategy(communication_options=communication_options, cluster_resolver=resolver
                                                         )  # communication_options=communication_options

    # ------------------------------------------
    # Preparing dataset
    # ------------------------------------------
    # Number of Machines use for Training
    per_gpu_train_batch = FLAGS.per_gpu_train_batch
    per_gpu_val_batch = FLAGS.per_gpu_val_batch

    train_global_batch_size = per_gpu_train_batch * strategy.num_replicas_in_sync
    val_global_batch_size = per_gpu_val_batch * strategy.num_replicas_in_sync

    dataset_loader = imagenet_dataset_multi_machine(img_size=FLAGS.image_size, train_batch=train_global_batch_size,
                                                    val_batch=val_global_batch_size,
                                                    strategy=strategy, train_path=FLAGS.train_path,
                                                    val_path=FLAGS.val_path,
                                                    mask_path=FLAGS.mask_path, bi_mask=True,
                                                    train_label=FLAGS.train_label, val_label=FLAGS.val_label,
                                                    subset_class_num=FLAGS.num_classes, subset_percentage=FLAGS.subset_percentage)
    ## Distributed data input Option
    input_options = tf.distribute.InputOptions(
        experimental_place_dataset_on_device = True,
        experimental_fetch_to_device = False,
        experimental_replication_mode = tf.distribute.InputReplicationMode.PER_REPLICA)
    train_multi_worker_dataset = strategy.distribute_datasets_from_function(
        lambda input_context: dataset_loader.simclr_random_global_crop_image_mask(input_context),input_options )

    val_multi_worker_dataset = strategy.distribute_datasets_from_function(
        lambda input_context: dataset_loader.supervised_validation(input_context), input_options)

    num_train_examples, num_eval_examples = dataset_loader.get_data_size()

    train_steps = FLAGS.eval_steps or int(
        num_train_examples * FLAGS.train_epochs // train_global_batch_size)*2
    eval_steps = FLAGS.eval_steps or int(
        math.ceil(num_eval_examples / val_global_batch_size))

    epoch_steps = int(round(num_train_examples / train_global_batch_size))
    checkpoint_steps = (FLAGS.checkpoint_steps or (
        FLAGS.checkpoint_epochs * epoch_steps))

    logging.info("# Subset_training class %d", FLAGS.num_classes)
    logging.info('# train examples: %d', num_train_examples)
    logging.info('# train_steps: %d', train_steps)
    logging.info('# eval examples: %d', num_eval_examples)
    logging.info('# eval steps: %d', eval_steps)

    with strategy.scope():
        online_model = all_model.Binary_online_model(
            FLAGS.num_classes, Upsample=FLAGS.feature_upsample, Downsample=FLAGS.downsample_mod)
        prediction_model = all_model.prediction_head_model()
        target_model = all_model.Binary_target_model(
            FLAGS.num_classes, Upsample=FLAGS.feature_upsample, Downsample=FLAGS.downsample_mod)
    # end first strategy

    # Configure Wandb Training
    # Weight&Bias Tracking Experiment
    configs = {
        "Model_Arch": "ResNet" + str(FLAGS.resnet_depth),
        "Training mode": "Baseline Non_Contrastive",
        "DataAugmentation_types": "SimCLR_Inception_style_Croping",
        "Dataset": "ImageNet1k",
        "IMG_SIZE": FLAGS.image_size,
        "Epochs": FLAGS.train_epochs,
        "Batch_size": train_global_batch_size,
        "Learning_rate": FLAGS.base_lr,
        "Temperature": FLAGS.temperature,
        "Optimizer": FLAGS.optimizer,
        "SEED": FLAGS.SEED,
        "Subset_dataset": FLAGS.num_classes,
        "Loss type": FLAGS.aggregate_loss,
        "opt": FLAGS.up_scale,
        "Encoder output size": str(list(FLAGS.Encoder_block_strides.values()).count(1) * 7),
    }

    wandb.init(project=FLAGS.wandb_project_name, name=FLAGS.wandb_run_name, mode=FLAGS.wandb_mod,
               sync_tensorboard=True, config=configs)

    # Training Configuration
    # *****************************************************************
    # Only Evaluate model
    # *****************************************************************

    if FLAGS.mode == "eval":
        # can choose different min_interval
        for ckpt in tf.train.checkpoints_iterator(FLAGS.model_dir, min_interval_secs=15):
            result = perform_evaluation(
                online_model, val_multi_worker_dataset, eval_steps, ckpt, strategy)
            # global_step from ckpt
            if result['global_step'] >= train_steps:
                logging.info('Evaluation complete. Existing-->')

    # *****************************************************************
    # Pre-Training and Evaluate
    # *****************************************************************
    else:
        summary_writer = tf.summary.create_file_writer(FLAGS.model_dir)

        with strategy.scope():

            # Configure the learning rate
            base_lr = FLAGS.base_lr
            scale_lr = FLAGS.lr_rate_scaling
            warmup_epochs = FLAGS.warmup_epochs
            train_epochs = FLAGS.train_epochs
            lr_schedule = WarmUpAndCosineDecay(
                base_lr, train_global_batch_size, num_train_examples, scale_lr, warmup_epochs,
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
            # Task type and Task_id among all Training Nodes
            task_type, task_id = (strategy.cluster_resolver.task_type,
                                  strategy.cluster_resolver.task_id)

            checkpoint_manager, write_checkpoint_dir = multi_node_try_restore_from_checkpoint(
                online_model, optimizer.iterations, optimizer, task_type, task_id)
            steps_per_loop = checkpoint_steps

            # Scale loss  --> Aggregating all Gradients

            # @tf.function
            def distributed_loss(o1, o2, b1, b2, f1=None, f2=None, alpha=0.5, weight=0.5):

                if FLAGS.non_contrast_binary_loss == 'original_add_backgroud':
                    ob1 = tf.concat([o1, b1], axis=0)
                    ob2 = tf.concat([o2, b2], axis=0)
                    # each GPU loss per_replica batch loss
                    per_example_loss, logits_ab, labels = byol_loss(
                        ob1, ob2, temperature=FLAGS.temperature)

                elif FLAGS.non_contrast_binary_loss == 'sum_symetrize_l2_loss_object_backg':

                    # each GPU loss per_replica batch loss
                    per_example_loss, logits_ab, labels = sum_symetrize_l2_loss_object_backg(
                        o1, o2, b1, b2, alpha=alpha, temperature=FLAGS.temperature)

                elif FLAGS.non_contrast_binary_loss == 'sum_symetrize_l2_loss_object_backg_add_original':
                    per_example_loss, logits_ab, labels = sum_symetrize_l2_loss_object_backg_add_original(
                        o1, o2, b1, b2, f1, f2, alpha=alpha, temperature=FLAGS.temperature, weight_loss=weight)
                else:
                    raise ValueError("Invalid Loss Type")
                # total sum loss //Global batch_size
                loss = 2 - 2*(tf.reduce_sum(per_example_loss)
                              * (1. / train_global_batch_size))
                # loss = tf.reduce_sum(per_example_loss) * \
                #     (1. / strategy.num_replicas_in_sync)

                return loss, logits_ab, labels

            @tf.function
            def train_step(ds_one, ds_two, alpha, weight_loss):
                # Get the data from
                images_mask_one, m11, m12, lable_1, = ds_one  # lable_one
                images_mask_two, m21, m22, lable_2, = ds_two  # lable_two

                '''
                Attention to Symetrize the loss --> Need to switch image_1, image_2 to (Online -- Target Network)
                loss 1= L2_loss*[online_model(image1), target_model(image_2)]
                loss 2=  L2_loss*[online_model(image2), target_model(image_1)]
                symetrize_loss= (loss 1+ loss_2)/ 2
                '''
                # Currently Our Loss function is Asymetrize L2_Loss
                with tf.GradientTape(persistent=True) as tape:

                    if FLAGS.loss_type == "symmetrized":

                        # Passing image 1, image 2 to Online Encoder , Target Encoder
                        # -------------------------------------------------------------
                        obj_1, backg_1, proj_head_output_1, supervised_head_output_1 = online_model(
                            [images_mask_one, m11, m12], training=True)
                        # Vector Representation from Online encoder go into Projection head again
                        obj_1 = prediction_model(obj_1, training=True)
                        backg_1 = prediction_model(backg_1, training=True)

                        proj_head_output_1 = prediction_model(
                            proj_head_output_1, training=True)

                        obj_2, backg_2, proj_head_output_2, supervised_head_output_2 = target_model(
                            [images_mask_two, m21, m22], training=True)

                        # -------------------------------------------------------------
                        # Passing Image 1, Image 2 to Target Encoder,  Online Encoder
                        # -------------------------------------------------------------
                        obj_2_online, backg_2_online, proj_head_output_2_online, _ = online_model(
                            [images_mask_two, m21, m22], training=True)
                        # Vector Representation from Online encoder go into Projection head again
                        obj_2_online = prediction_model(
                            obj_2_online, training=True)
                        backg_2_online = prediction_model(
                            backg_2_online, training=True)

                        proj_head_output_2_online = prediction_model(
                            proj_head_output_2_online, training=True)

                        obj_1_target, backg_1_target, proj_head_output_1_target, _ = \
                            target_model(
                                [images_mask_one, m11, m12], training=True)

                        # Compute Contrastive Train Loss -->
                        loss = None
                        if proj_head_output_1 is not None:
                            # Loss of the image 1, 2 --> Online, Target Encoder
                            loss_1, logits_o_ab, labels = distributed_loss(
                                obj_1, obj_2, backg_1, backg_2, proj_head_output_1, proj_head_output_2, alpha,
                                weight_loss)

                            # Loss of the image 2, 1 --> Online, Target Encoder
                            loss_2, logits_o_ab_2, labels_2 = distributed_loss(
                                obj_2_online, obj_1_target, backg_2_online, backg_1_target, proj_head_output_2_online,
                                proj_head_output_1_target, alpha, weight_loss)

                            # Total loss
                            loss = (loss_1 + loss_2) / 2

                            if loss is None:
                                loss = loss
                            else:
                                loss += loss

                            # Update Self-Supervised Metrics
                            metrics.update_pretrain_metrics_train_multi_machine(contrast_loss_metric,
                                                                                contrast_acc_metric,
                                                                                contrast_entropy_metric,
                                                                                loss, logits_o_ab,
                                                                                labels)

                    elif FLAGS.loss_type == "asymmetrized":
                        obj_1, backg_1, proj_head_output_1, supervised_head_output_1 = online_model(
                            [images_mask_one, m11, m12], training=True)
                        # Vector Representation from Online encoder go into Projection head again
                        obj_1 = prediction_model(obj_1, training=True)
                        backg_1 = prediction_model(backg_1, training=True)
                        proj_head_output_1 = prediction_model(
                            proj_head_output_1, training=True)

                        obj_2, backg_2, proj_head_output_2, supervised_head_output_2 = target_model(
                            [images_mask_two, m21, m22], training=True)

                        # Compute Contrastive Train Loss -->
                        loss = None
                        if proj_head_output_1 is not None:
                            loss, logits_o_ab, labels = distributed_loss(
                                obj_1, obj_2, backg_1, backg_2, proj_head_output_1, proj_head_output_2, alpha,
                                weight_loss)

                            if loss is None:
                                loss = loss
                            else:
                                loss += loss

                            # Update Self-Supervised Metrics
                            metrics.update_pretrain_metrics_train_multi_machine(contrast_loss_metric,
                                                                                contrast_acc_metric,
                                                                                contrast_entropy_metric,
                                                                                loss, logits_o_ab,
                                                                                labels)

                    else:
                        raise ValueError(
                            'invalid loss type check your loss type')

                    # Compute the Supervised train Loss
                    '''Consider Sperate Supervised Loss'''
                    # supervised_loss=None
                    if supervised_head_output_1 is not None:
                        if FLAGS.train_mode == 'pretrain' and FLAGS.lineareval_while_pretraining:
                            outputs = tf.concat(
                                [supervised_head_output_1, supervised_head_output_2], 0)
                            supervise_lable = tf.concat(
                                [lable_1, lable_2], 0)

                            # Calculte the cross_entropy loss with Labels
                            sup_loss = obj_lib.add_supervised_loss(
                                labels=supervise_lable, logits=outputs)

                            scale_sup_loss = tf.nn.compute_average_loss(
                                sup_loss, global_batch_size=train_global_batch_size)
                            # scale_sup_loss = tf.reduce_sum(
                            #     sup_loss) * (1./train_global_batch)
                            # Update Supervised Metrics
                            metrics.update_finetune_metrics_train(supervised_loss_metric,
                                                                  supervised_acc_metric, scale_sup_loss,
                                                                  supervise_lable, outputs)

                        '''Attention'''
                        # Noted Consideration Aggregate (Supervised + Contrastive Loss) --> Update the Model Gradient
                        # if FLAGS.precision_method == "API":
                        #     scale_sup_loss = tf.cast(scale_sup_loss, 'float16')

                        if FLAGS.aggregate_loss == "contrastive_supervised":
                            if loss is None:
                                loss = scale_sup_loss
                            else:
                                loss += scale_sup_loss
                        elif FLAGS.aggregate_loss == "contrastive":

                            supervise_loss = None
                            if supervise_loss is None:
                                supervise_loss = scale_sup_loss
                            else:
                                supervise_loss += scale_sup_loss
                        else:
                            raise ValueError(
                                " Loss aggregate is invalid please check FLAGS.aggregate_loss")

                    weight_decay_loss = all_model.add_weight_decay(
                        online_model, adjust_per_optimizer=True)

                    # Under experiment Scale loss after adding Regularization and scaled by Batch_size
                    # weight_decay_loss = tf.nn.scale_regularization_loss(
                    #     weight_decay_loss)

                    weight_decay_metric.update_state(weight_decay_loss)
                    # if FLAGS.precision_method == "API":
                    #     weight_decay_loss = tf.cast(
                    #         weight_decay_loss, 'float16')
                    loss += weight_decay_loss
                    total_loss_metric.update_state(loss)

                    logging.info('Trainable variables:')
                    for var in online_model.trainable_variables:
                        logging.info(var.name)

                if FLAGS.mixprecision == "fp16":
                    logging.info("you implement mix_percision_16_Fp")

                    # Method 1
                    if FLAGS.precision_method == "API":
                        # Reduce loss Precision to 16 Bits
                        scaled_loss = optimizer.get_scaled_loss(loss)
                        # Update the Encoder
                        scaled_gradients = tape.gradient(
                            scaled_loss, online_model.trainable_variables)
                        # all_reduce_fp16_grads_online = tf.distribute.get_replica_context(
                        # ).all_reduce(tf.distribute.ReduceOp.SUM, scaled_gradients)

                        gradients_online = optimizer.get_unscaled_gradients(
                            scaled_gradients)
                        optimizer.apply_gradients(zip(
                            gradients_online, online_model.trainable_variables), )

                        # Update Prediction Head model
                        scaled_grads_pred = tape.gradient(
                            scaled_loss, prediction_model.trainable_variables)
                        # all_reduce_fp16_grads_pred = tf.distribute.get_replica_context(
                        # ).all_reduce(tf.distribute.ReduceOp.SUM, scaled_grads_pred)

                        gradients_pred = optimizer.get_unscaled_gradients(
                            scaled_grads_pred)
                        optimizer.apply_gradients(
                            zip(gradients_pred, prediction_model.trainable_variables), )

                    # Method 2
                    if FLAGS.precision_method == "custome":

                        # Online model
                        grads_online = tape.gradient(
                            loss, online_model.trainable_variables)
                        fp16_grads_online = [
                            tf.cast(grad, 'float16')for grad in grads_online]

                        # Optional
                        if FLAGS.collective_hint:
                            hints = tf.distribute.experimental.CollectiveHints(
                                bytes_per_pack=32 * 1024 * 1024)
                            all_reduce_fp16_grads_online = tf.distribute.get_replica_context().all_reduce(
                                tf.distribute.ReduceOp.SUM, fp16_grads_online, options=hints)
                        else:
                            all_reduce_fp16_grads_online = tf.distribute.get_replica_context(
                            ).all_reduce(tf.distribute.ReduceOp.SUM, fp16_grads_online)

                        all_reduce_fp32_grads_online = [
                            tf.cast(grad, 'float32') for grad in all_reduce_fp16_grads_online]
                        # all_reduce_fp32_grads_online = optimizer.get_unscaled_gradients(
                        #     all_reduce_fp16_grads_online)
                        # all_reduce_fp32_grads = optimizer.get_unscaled_gradients(
                        #     all_reduce_fp32_grads)
                        optimizer.apply_gradients(zip(
                            all_reduce_fp32_grads_online, online_model.trainable_variables), experimental_aggregate_gradients=False)

                        # Prediction Model
                        grads_pred = tape.gradient(
                            loss, prediction_model.trainable_variables)
                        fp16_grads_pred = [
                            tf.cast(grad, 'float16')for grad in grads_pred]

                        if FLAGS.collective_hint:
                            hints = tf.distribute.experimental.CollectiveHints(
                                bytes_per_pack=32 * 1024 * 1024)
                            all_reduce_fp16_grads_pred = tf.distribute.get_replica_context().all_reduce(
                                tf.distribute.ReduceOp.SUM, fp16_grads_pred, options=hints)
                        else:
                            all_reduce_fp16_grads_pred = tf.distribute.get_replica_context(
                            ).all_reduce(tf.distribute.ReduceOp.SUM, fp16_grads_pred)

                        all_reduce_fp32_grads_pred = [
                            tf.cast(grad, 'float32') for grad in all_reduce_fp16_grads_pred]

                        # all_reduce_fp32_grads = optimizer.get_unscaled_gradients(
                        #     all_reduce_fp32_grads)
                        optimizer.apply_gradients(zip(
                            all_reduce_fp32_grads_pred, prediction_model.trainable_variables), experimental_aggregate_gradients=False)

                elif FLAGS.mixprecision == "fp32":
                    logging.info("you implement original_Fp precision")

                    # Update Encoder and Projection head weight
                    grads_online = tape.gradient(
                        loss, online_model.trainable_variables)
                    if FLAGS.collective_hint:
                        hints = tf.distribute.experimental.CollectiveHints(
                            bytes_per_pack=50 * 1024 * 1024)
                        # options = tf.distribute.experimental.CommunicationOptions(
                        #     bytes_per_pack=50 * 1024 * 1024,
                        #     timeout_seconds=120.0,
                        #     implementation=tf.distribute.experimental.CommunicationImplementation.NCCL
                        # )
                        grads_online = tf.distribute.get_replica_context().all_reduce(
                            tf.distribute.ReduceOp.SUM, grads_online, options=hints)
                    else:
                        grads_online = tf.distribute.get_replica_context().all_reduce(
                            tf.distribute.ReduceOp.SUM, grads_online)
                        print("local_grad")
                    optimizer.apply_gradients(
                        zip(grads_online, online_model.trainable_variables), experimental_aggregate_gradients=False)

                    # Update Prediction Head model
                    grads_pred = tape.gradient(
                        loss, prediction_model.trainable_variables)

                    if FLAGS.collective_hint:
                        hints = tf.distribute.experimental.CollectiveHints(
                            bytes_per_pack=50 * 1024 * 1024)
                        # options = tf.distribute.experimental.CommunicationOptions(
                        #     bytes_per_pack=50 * 1024 * 1024,
                        #     timeout_seconds=120.0,
                        #     implementation=tf.distribute.experimental.CommunicationImplementation.NCCL
                        # )

                        grads_pred = tf.distribute.get_replica_context().all_reduce(
                            tf.distribute.ReduceOp.SUM, grads_pred, options=hints)
                    else:
                        print("local_grad")
                        grads_pred = tf.distribute.get_replica_context().all_reduce(
                            tf.distribute.ReduceOp.SUM, grads_pred)

                    optimizer.apply_gradients(
                        zip(grads_pred, prediction_model.trainable_variables), experimental_aggregate_gradients=False)  # all_reduce_sum_gradients=FalseUpdate gradient Customize
                else:
                    raise ValueError(
                        "Invalid Implement optimization floating precision")
                del tape
                return loss

            @tf.function
            def distributed_train_step(ds_one, ds_two, alpha, weight_loss):
                per_replica_losses = strategy.run(
                    train_step, args=(ds_one, ds_two, alpha, weight_loss))
                return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                       axis=None)
            global_step = optimizer.iterations

            # Train the model here
            # tf.profiler.experimental.start(FLAGS.model_dir)

            for epoch in range(FLAGS.train_epochs):
                total_loss = 0.0
                num_batches = 0
                print("Epoch", epoch, "...")
                for step, (ds_one, ds_two) in enumerate(train_multi_worker_dataset):

                    # Update Two different Alpha Schedule for increasing Values
                    if FLAGS.alpha_schedule == "cosine_schedule":
                        logging.info(
                            "Implementation beta momentum uses Cosine Function")
                        alpha_base = 0.5
                        cur_step = float(global_step.numpy())
                        alpha = 1 - (1 - alpha_base) * \
                            (math.cos(math.pi * cur_step / train_steps) + 1) / 2

                    if FLAGS.alpha_schedule == "custom_schedule":
                        if epoch + 1 <= 0.7 * FLAGS.train_epochs:
                            alpha = 0.5
                            # weight_loss = 0.5
                        elif epoch + 1 <= 0.9 * FLAGS.train_epochs:
                            alpha = 0.7
                            # weight_loss = 0.7
                        else:
                            alpha = 0.9

                    total_loss += distributed_train_step(
                        ds_one, ds_two, alpha, FLAGS.weighted_loss)
                    num_batches += 1

                    # Update weight of Target Encoder Every Step
                    if FLAGS.moving_average == "fixed_value":
                        beta = 0.99
                    elif FLAGS.moving_average == "schedule":
                        # This update the Beta value schedule along with Trainign steps Follow BYOL
                        logging.info(
                            "Implementation beta momentum uses Cosine Function")
                        beta_base = 0.996
                        cur_step = global_step.numpy()
                        beta = 1 - (1 - beta_base) * \
                            (math.cos(math.pi * cur_step / train_steps) + 1) / 2
                    else:
                        raise ValueError("Invalid Option of Moving average")

                    target_encoder_weights = target_model.get_weights()
                    online_encoder_weights = online_model.get_weights()
                    for i in range(len(online_encoder_weights)):
                        target_encoder_weights[i] = beta * target_encoder_weights[i] + (
                            1 - beta) * online_encoder_weights[i]
                    target_model.set_weights(target_encoder_weights)

                    # if step == 10 and epoch == 1:
                    #     tf.profiler.experimental.start(FLAGS.model_dir)
                    # if step == 30 and epoch == 1:
                    #     print("stop profile")
                    #     tf.profiler.experimental.stop()

                    with summary_writer.as_default():
                        cur_step = global_step.numpy()
                        checkpoint_manager.save(cur_step)
                        # Removing the checkpoint if it is not Chief Worker
                        if not chief_worker(task_type, task_id):
                            tf.io.gfile.rmtree(write_checkpoint_dir)

                        logging.info('Completed: %d / %d steps',
                                     cur_step, train_steps)
                        metrics.log_and_write_metrics_to_summary(
                            all_metrics, cur_step)
                        tf.summary.scalar('learning_rate', lr_schedule(tf.cast(global_step, dtype=tf.float32)),
                                          global_step)
                        summary_writer.flush()

                epoch_loss = total_loss / num_batches
                # Configure for Visualize the Model Training
                if (epoch + 1) % 10 == 0:
                    FLAGS.train_mode = 'finetune'
                    result = perform_evaluation(online_model, val_multi_worker_dataset, eval_steps,
                                                checkpoint_manager.latest_checkpoint, strategy)
                    wandb.log({
                        "eval/label_top_1_accuracy": result["eval/label_top_1_accuracy"],
                        "eval/label_top_5_accuracy": result["eval/label_top_5_accuracy"],
                    })
                    FLAGS.train_mode = 'pretrain'

                wandb.log({
                    "epochs": epoch + 1,
                    "train/alpha_value": alpha,
                    "train/weight_loss_value": FLAGS.weighted_loss,
                    "train_contrast_loss": contrast_loss_metric.result(),
                    "train_contrast_acc": contrast_acc_metric.result(),
                    "train_contrast_acc_entropy": contrast_entropy_metric.result(),
                    "train/weight_decay": weight_decay_metric.result(),
                    "train/total_loss": epoch_loss,
                    "train/supervised_loss": supervised_loss_metric.result(),
                    "train/supervised_acc": supervised_acc_metric.result(),
                })

                for metric in all_metrics:
                    metric.reset_states()
                # Saving Entire Model

                if (epoch + 1) % 2 == 0:
                    save_encoder = os.path.join(
                        FLAGS.model_dir, "encoder_model_" + str(epoch) + ".h5")
                    save_online_model = os.path.join(
                        FLAGS.model_dir, "online_model_" + str(epoch) + ".h5")
                    save_target_model = os.path.join(
                        FLAGS.model_dir, "target_model_" + str(epoch) + ".h5")
                    online_model.encoder.save_weights(save_encoder)
                    online_model.save_weights(save_online_model)
                    target_model.save_weights(save_target_model)

            logging.info('Training Complete ...')

        # end second strategy

        if FLAGS.mode == 'train_then_eval':
            perform_evaluation(online_model, val_multi_worker_dataset, eval_steps,
                               checkpoint_manager.latest_checkpoint, strategy)

        save_encoder = os.path.join(
            FLAGS.model_dir, "encoder_model_latest.h5")
        save_online_model = os.path.join(
            FLAGS.model_dir, "online_model_latest.h5")
        save_target_model = os.path.join(
            FLAGS.model_dir, "target_model_latest.h5")
        online_model.resnet_model.save_weights(save_encoder)
        online_model.save_weights(save_online_model)
        target_model.save_weights(save_target_model)

# # Restore model weights only, but not global step and optimizer states
# flags.DEFINE_string(
#     'checkpoint', None,
#     'Loading from the given checkpoint for fine-tuning if a finetuning '
#     'checkpoint does not already exist in model_dir.')

    # Pre-Training and Finetune
if __name__ == '__main__':

    main()
