from config.experiment_config import read_cfg
#from config.config_for_add_orgloss import read_cfg
from config.absl_mock import Mock_Flag
import json
import math
import wandb
import random
# from absl import flags
from absl import logging
# from absl import app

import tensorflow as tf
from learning_rate_optimizer import WarmUpAndCosineDecay, CosineAnnealingDecayRestarts
import metrics
from helper_functions import *
from byol_simclr_imagenet_data_harry import imagenet_dataset_single_machine
from self_supervised_losses import byol_symetrize_loss, symetrize_l2_loss_object_level_whole_image, sum_symetrize_l2_loss_object_backg, sum_symetrize_l2_loss_object_backg_add_original
import model_for_non_contrastive_framework as all_model
import objective as obj_lib
from imutils import paths
from wandb.keras import WandbCallback

# Setting GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices(gpus[0:8], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        print(e)

read_cfg()
flag = Mock_Flag()
FLAGS = flag.FLAGS
if not os.path.isdir(FLAGS.model_dir):
    print("Creat the model dir: ",FLAGS.model_dir)
    os.makedirs(FLAGS.model_dir)
flag.save_config(os.path.join(FLAGS.model_dir,"config.cfg"))


def main():
    # Preparing dataset
    # Imagenet path prepare localy
    strategy = tf.distribute.MirroredStrategy()
    train_global_batch = FLAGS.train_batch_size * strategy.num_replicas_in_sync
    val_global_batch = FLAGS.val_batch_size * strategy.num_replicas_in_sync

    train_dataset = imagenet_dataset_single_machine(img_size=FLAGS.image_size, train_batch=train_global_batch,  val_batch=val_global_batch,
                                                    strategy=strategy, train_path=FLAGS.train_path,
                                                    val_path=FLAGS.val_path,
                                                    mask_path=FLAGS.mask_path, bi_mask=True,
                                                    train_label=FLAGS.train_label, val_label=FLAGS.val_label,
                                                    subset_class_num=FLAGS.num_classes)


    val_ds = train_dataset.supervised_validation()

    num_train_examples, num_eval_examples = train_dataset.get_data_size()

    train_steps = FLAGS.eval_steps or int(
        num_train_examples * FLAGS.train_epochs // train_global_batch)*2

    eval_steps = FLAGS.eval_steps or int(
        math.ceil(num_eval_examples / val_global_batch))

    epoch_steps = int(round(num_train_examples / train_global_batch))

    checkpoint_steps = (FLAGS.checkpoint_steps or (
        FLAGS.checkpoint_epochs * epoch_steps))

    logging.info("# Subset_training class %d", FLAGS.num_classes)
    logging.info('# train examples: %d', num_train_examples)
    logging.info('# train_steps: %d', train_steps)
    logging.info('# eval examples: %d', num_eval_examples)
    logging.info('# eval steps: %d', eval_steps)

    # Configure the Encoder Architecture.
    with strategy.scope():
        online_model = all_model.online_model(FLAGS.num_classes)

    # Configure Wandb Training
    # Weight&Bias Tracking Experiment
    configs = {

        "Model_Arch": "ResNet50",
        "Training mode": "Binary_Non_Contrative_SSL",
        "DataAugmentation_types": "SimCLR_Inception_Croping_image_mask",
        "Speratation Features Upsampling Method": FLAGS.feature_upsample,
        "Dataset": "ImageNet1k",
        "object_backgroud_feature_Dsamp_method": FLAGS.downsample_mod,

        "IMG_SIZE": FLAGS.image_size,
        "Epochs": FLAGS.train_epochs,
        "Batch_size": train_global_batch,
        "Learning_rate": FLAGS.base_lr,
        "Temperature": FLAGS.temperature,
        "Optimizer": FLAGS.optimizer,
        "SEED": FLAGS.SEED,
        "Subset_dataset": FLAGS.num_classes,

        "Loss configure": FLAGS.aggregate_loss,
        "Loss type": FLAGS.non_contrast_binary_loss,

    }

    wandb.init(project=FLAGS.wandb_project_name, name=FLAGS.wandb_run_name, mode=FLAGS.wandb_mod,
               sync_tensorboard=True, config=configs)

    # Training Configuration
    # *****************************************************************
    # Only Evaluate model
    # *****************************************************************
    online_model.built = True
    online_model.build((1, 224, 224, 3))
    online_model.load_weights("/data1/share/resnet_byol/restnet50/Baseline_(7_7_2048)_200epoch/online_model_199.h5")
    if FLAGS.mode == "eval":
        # can choose different min_interval
        for ckpt in tf.train.checkpoints_iterator(FLAGS.model_dir, min_interval_secs=15):
            result = perform_evaluation(online_model, val_ds, eval_steps, ckpt, strategy)
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
            if FLAGS.lr_strategies == "warmup_cos_lr":
                base_lr = FLAGS.base_lr
                scale_lr = FLAGS.lr_rate_scaling
                warmup_epochs = FLAGS.warmup_epochs
                train_epochs = FLAGS.train_epochs

                lr_schedule = WarmUpAndCosineDecay(
                    base_lr, train_global_batch, num_train_examples, scale_lr, warmup_epochs,
                    train_epochs=train_epochs, train_steps=train_steps)

            elif FLAGS.lr_strategies == "cos_annealing_restart":
                base_lr = FLAGS.base_lr
                scale_lr = FLAGS.lr_rate_scaling
                # Control cycle of next step base of Previous step (2 times more steps)
                t_mul = 2.0
                # Control ititial Learning Rate Values (Next step equal to previous steps)
                m_mul = 1.0
                alpha = 0.0  # Final values of learning rate
                first_decay_steps = train_steps / (FLAGS.number_cycles * t_mul)
                lr_schedule = CosineAnnealingDecayRestarts(
                    base_lr, first_decay_steps, train_global_batch, scale_lr, t_mul=t_mul, m_mul=m_mul, alpha=alpha)

            # Current Implement the Mixpercision optimizer
            optimizer = all_model.build_optimizer(lr_schedule)

            # Build tracking metrics
            all_metrics = []
            # Linear classfiy metric
            weight_decay_metric = tf.keras.metrics.Mean('train/weight_decay')
            total_loss_metric = tf.keras.metrics.Mean('train/total_loss')
            all_metrics.extend([weight_decay_metric, total_loss_metric])

            logging.info(
                "Apllying pre-training and Linear evaluation at the same time")
            # Fine-tune architecture metrics
            supervised_loss_metric = tf.keras.metrics.Mean(
                'train/supervised_loss')
            supervised_acc_metric = tf.keras.metrics.Mean(
                'train/supervised_acc')
            all_metrics.extend(
                [supervised_loss_metric, supervised_acc_metric])

            @tf.function
            def train_step(ds):

                # Get the data from
                images, lable_1, = ds

                with tf.GradientTape(persistent=True) as tape:

                    _, _,  _, supervised_head_output_1 = online_model(
                        [images_mask_one[0], tf.expand_dims(images_mask_one[1], axis=-1)], training=True)
                    # Vector Representation from Online encoder go into Projection head again
                    # Compute Contrastive Train Loss -->
                    if supervised_head_output_1 is not None:
                        outputs = tf.concat([supervised_head_output_1, supervised_head_output_2], 0)
                        supervise_lable = tf.concat([lable_1, lable_2], 0)

                        # Calculte the cross_entropy loss with Labels
                        sup_loss = obj_lib.add_supervised_loss(
                            labels=supervise_lable, logits=outputs)

                        scale_sup_loss = tf.nn.compute_average_loss(
                            sup_loss, global_batch_size=train_global_batch)

                        metrics.update_finetune_metrics_train(supervised_loss_metric,
                                                              supervised_acc_metric, scale_sup_loss,
                                                              supervise_lable, outputs)
                    loss = scale_sup_loss

                    weight_decay_loss = all_model.add_weight_decay(
                        online_model, adjust_per_optimizer=True)

                    weight_decay_metric.update_state(weight_decay_loss)
                    loss += weight_decay_loss
                    total_loss_metric.update_state(loss)

                    logging.info('Trainable variables:')
                    for var in online_model.trainable_variables:
                        logging.info(var.name)

                # Update Encoder and Projection head weight
                grads = tape.gradient(loss, online_model.trainable_variables)
                optimizer.apply_gradients(zip(grads, online_model.trainable_variables))
                del tape
                return loss

            @tf.function
            def distributed_train_step(ds):
                per_replica_losses = strategy.run(
                    train_step, args=(ds))
                return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                       axis=None)
            global_step = optimizer.iterations

            for epoch in range(FLAGS.train_epochs):
                total_loss = 0.0
                num_batches = 0
                # alpha=1.0
                weight_loss=FLAGS.weighted_loss
                print("Epoch", epoch, "...")
                for _, ds in enumerate(val_ds):
                    total_loss += distributed_train_step(ds)
                    num_batches += 1

                    with summary_writer.as_default():
                        cur_step = global_step.numpy()
                        checkpoint_manager.save(cur_step)
                        logging.info('Completed: %d / %d steps',
                                     cur_step, train_steps)
                        metrics.log_and_write_metrics_to_summary(
                            all_metrics, cur_step)
                        tf.summary.scalar('learning_rate', lr_schedule(tf.cast(global_step, dtype=tf.float32)),
                                          global_step)
                        summary_writer.flush()

                epoch_loss = total_loss/num_batches
                # Wandb Configure for Visualize the Model Training

                for metric in all_metrics:
                    metric.reset_states()
                # Saving Entire Model
                if (epoch+1) % 20 == 0:
                    save_encoder = os.path.join(
                        FLAGS.model_dir, "encoder_model_" + str(epoch) + ".h5")
                    save_online_model = os.path.join(
                        FLAGS.model_dir, "online_model_" + str(epoch) + ".h5")
                    online_model.encoder.save_weights(save_encoder)
                    online_model.save_weights(save_online_model)

                result = perform_evaluation(online_model, val_ds, eval_steps, ckpt, strategy)
                wandb.log({
                    "epochs": epoch+1,
                    "train/alpha_value": alpha,
                    "train/weight_loss_value": weight_loss,
                    "train_contrast_loss": contrast_loss_metric.result(),
                    "train_contrast_acc": contrast_acc_metric.result(),
                    "train_contrast_acc_entropy": contrast_entropy_metric.result(),
                    "train/weight_decay": weight_decay_metric.result(),
                    "train/total_loss": epoch_loss,
                    "train/supervised_loss": supervised_loss_metric.result(),
                    "train/supervised_acc": supervised_acc_metric.result(),
                    "eval/label_top_1_accuracy":result["eval/label_top_1_accuracy"],
                    "eval/label_top_5_accuracy": result["eval/label_top_5_accuracy"],
                })

            logging.info('Training Complete ...')


        if FLAGS.mode == 'train_then_eval':
            perform_evaluation(online_model, val_ds, eval_steps,
                               checkpoint_manager.latest_checkpoint, strategy)

        save_encoder = os.path.join(
            FLAGS.model_dir, "encoder_model_latest.h5")
        save_online_model = os.path.join(
            FLAGS.model_dir, "online_model_latest.h5")
        online_model.resnet_model.save_weights(save_encoder)
        online_model.save_weights(save_online_model)

    # Pre-Training and Finetune
if __name__ == '__main__':
    main()
