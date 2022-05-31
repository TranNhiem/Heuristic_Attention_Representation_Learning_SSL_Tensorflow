from math import cos, pi

from config.experiment_config import read_cfg

import wandb
from HARL.utils.learning_rate_optimizer import WarmUpAndCosineDecay, CosineAnnealingDecayRestarts
from HARL.utils.helper_functions import *
from HARL.DataAugmentations.byol_simclr_imagenet_data_harry import imagenet_dataset_single_machine
from HARL.loss.self_supervised_losses import byol_symetrize_loss, symetrize_l2_loss_object_level_whole_image, \
    sum_symetrize_l2_loss_object_backg, sum_symetrize_l2_loss_object_backg_add_original
import model_for_non_contrastive_framework as all_model
import HARL.loss.objective as obj_lib

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
    print("Creat the model dir: ", FLAGS.model_dir)
    os.makedirs(FLAGS.model_dir)
flag.save_config(os.path.join(FLAGS.model_dir, "config.cfg"))

# For setting GPUs Thread reduce kernel Luanch Delay
# https://github.com/tensorflow/tensorflow/issues/25724
# os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
# os.environ['TF_GPU_THREAD_COUNT'] = '1'

def main():
    # Preparing dataset
    # Imagenet path prepare localy
    strategy = tf.distribute.MirroredStrategy()
    train_global_batch = FLAGS.train_batch_size * strategy.num_replicas_in_sync
    val_global_batch = FLAGS.val_batch_size * strategy.num_replicas_in_sync

    train_dataset = imagenet_dataset_single_machine(img_size=FLAGS.image_size, train_batch=train_global_batch,
                                                    val_batch=val_global_batch,
                                                    strategy=strategy, train_path=FLAGS.train_path,
                                                    val_path=FLAGS.val_path,
                                                    mask_path=FLAGS.mask_path, bi_mask=True,
                                                    train_label=FLAGS.train_label, val_label=FLAGS.val_label,
                                                    subset_class_num=FLAGS.num_classes,subset_percentage=FLAGS.subset_percentage)

    train_ds = train_dataset.simclr_inception_style_crop_image_mask()

    val_ds = train_dataset.supervised_validation()

    num_train_examples, num_eval_examples = train_dataset.get_data_size()

    train_steps = FLAGS.eval_steps or int(
        num_train_examples * FLAGS.train_epochs // train_global_batch) * 2

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
        online_model = all_model.Binary_online_model(
            FLAGS.num_classes, Upsample=FLAGS.feature_upsample, Downsample=FLAGS.downsample_mod)
        prediction_model = all_model.prediction_head_model()
        target_model = all_model.Binary_target_model(
            FLAGS.num_classes, Upsample=FLAGS.feature_upsample, Downsample=FLAGS.downsample_mod)

    # Configure Wandb Training
    # Weight&Bias Tracking Experiment
    configs = {
        "Model_Arch": "ResNet" + str(FLAGS.resnet_depth),
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
        "Subset_dataset": f"Class : {FLAGS.num_classes}, Percentage : {FLAGS.subset_percentage*100}%",
        "Loss configure": FLAGS.aggregate_loss,
        "Loss type": FLAGS.non_contrast_binary_loss,
        "Encoder output size" : str((math.pow(2,list(FLAGS.Encoder_block_strides.values()).count(1))-1) * 7),
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
                online_model, val_ds, eval_steps, ckpt, strategy)
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

            # Scale loss  --> Aggregating all Gradients
            def distributed_loss(o1, o2, b1, b2, f1=None, f2=None, alpha=0.5, weight=0.5):
                per_example_loss = 0

                if FLAGS.non_contrast_binary_loss == 'original_add_backgroud':
                    ob1 = tf.concat([o1, b1], axis=0)
                    ob2 = tf.concat([o2, b2], axis=0)
                    # each GPU loss per_replica batch loss
                    per_example_loss, logits_ab, labels = byol_symetrize_loss(
                        ob1, ob2, temperature=FLAGS.temperature)

                elif FLAGS.non_contrast_binary_loss == 'sum_symetrize_l2_loss_object_backg':

                    # each GPU loss per_replica batch loss
                    per_example_loss, logits_ab, labels = sum_symetrize_l2_loss_object_backg(
                        o1, o2, b1, b2, alpha=alpha, temperature=FLAGS.temperature)

                elif FLAGS.non_contrast_binary_loss == 'sum_symetrize_l2_loss_object_backg_add_original':
                    per_example_loss, logits_ab, labels = sum_symetrize_l2_loss_object_backg_add_original(
                        o1, o2, b1, b2, f1, f2, alpha=alpha, temperature=FLAGS.temperature, weight_loss=weight)

                # total sum loss //Global batch_size
                loss = tf.reduce_sum(per_example_loss) * (1. / len(gpus))
                # loss = 2-2*loss
                

                return loss, logits_ab, labels

            def train_step(ds_one, ds_two, alpha, weight_loss):

                # Get the data from
                images_mask_one ,lable_1, = ds_one  # lable_one
                images_mask_two ,lable_2, = ds_two  # lable_two

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
                                images_mask_one, training=True)
                            # Vector Representation from Online encoder go into Projection head again
                            obj_1 = prediction_model(obj_1, training=True)
                            backg_1 = prediction_model(backg_1, training=True)

                            proj_head_output_1 = prediction_model(
                                proj_head_output_1, training=True)

                            obj_2, backg_2, proj_head_output_2, supervised_head_output_2 = target_model(
                                images_mask_two, training=True)

                            # -------------------------------------------------------------
                            # Passing Image 1, Image 2 to Target Encoder,  Online Encoder
                            # -------------------------------------------------------------
                            obj_2_online, backg_2_online, proj_head_output_2_online, _ = online_model(
                                images_mask_two, training=True)
                            # Vector Representation from Online encoder go into Projection head again
                            obj_2_online = prediction_model(
                                obj_2_online, training=True)
                            backg_2_online = prediction_model(
                                backg_2_online, training=True)

                            proj_head_output_2_online = prediction_model(
                                proj_head_output_2_online, training=True)

                            obj_1_target, backg_1_target, proj_head_output_1_target, _ = \
                                target_model(images_mask_one, training=True)

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
                                metrics.update_pretrain_metrics_train(contrast_loss_metric,
                                                                      contrast_acc_metric,
                                                                      contrast_entropy_metric,
                                                                      loss, logits_o_ab,
                                                                      labels)

                    elif FLAGS.loss_type == "asymmetrized":
                        obj_1, backg_1, proj_head_output_1, supervised_head_output_1 = online_model(
                            images_mask_one, training=True)
                        # Vector Representation from Online encoder go into Projection head again
                        obj_1 = prediction_model(obj_1, training=True)
                        backg_1 = prediction_model(backg_1, training=True)
                        proj_head_output_1 = prediction_model(
                            proj_head_output_1, training=True)

                        obj_2, backg_2, proj_head_output_2, supervised_head_output_2 = target_model(
                            images_mask_two, training=True)

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
                            metrics.update_pretrain_metrics_train(contrast_loss_metric,
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
                                sup_loss, global_batch_size=train_global_batch)
                            # scale_sup_loss = tf.reduce_sum(
                            #     sup_loss) * (1./train_global_batch)
                            # Update Supervised Metrics
                            metrics.update_finetune_metrics_train(supervised_loss_metric,
                                                                  supervised_acc_metric, scale_sup_loss,
                                                                  supervise_lable, outputs)

                    '''Attention'''
                    # Noted Consideration Aggregate (Supervised + Contrastive Loss) --> Update the Model Gradient
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

                    weight_decay_metric.update_state(weight_decay_loss)

                    # if FLAGS.mixprecision == "fp16":
                    #     #Casting the weight Decay loss also
                    #     weight_decay_loss=tf.cast(weight_decay_loss, 'float16')

                    loss += weight_decay_loss
                    total_loss_metric.update_state(loss)

                    logging.info('Trainable variables:')
                    for var in online_model.trainable_variables:
                        logging.info(var.name)

                if FLAGS.mixprecision == "fp16":
                    logging.info("you implement mix_percision_16_Fp")

                    # Method 1
                    # # Reduce loss Precision to 16 Bits
                    # scaled_loss = optimizer.get_scaled_loss(loss)

                    # # Update the Encoder
                    # scaled_gradients = tape.gradient(
                    #     scaled_loss, online_model.trainable_variables)
                    # gradients = optimizer.get_unscaled_gradients(scaled_gradients)
                    # optimizer.apply_gradients(
                    #     zip(gradients, online_model.trainable_variables))

                    # # Update Prediction Head model
                    # scaled_grads = tape.gradient(
                    #     scaled_loss, prediction_model.trainable_variables)
                    # gradients_unscale = optimizer.get_unscaled_gradients(scaled_grads)
                    # optimizer.apply_gradients(
                    #     zip(gradients_unscale, prediction_model.trainable_variables))

                    # Method 2
                    fp32_grads = tape.gradient(
                        loss, online_model.trainable_variables)
                    fp16_grads = [tf.cast(grad, 'float16')
                                  for grad in fp32_grads]
                    all_reduce_fp16_grads = tf.distribute.get_replica_context(
                    ).all_reduce(tf.distribute.ReduceOp.SUM, fp16_grads)
                    all_reduce_fp32_grads = [
                        tf.cast(grad, 'float32') for grad in all_reduce_fp16_grads]

                    # all_reduce_fp32_grads = optimizer.get_unscaled_gradients(
                    #     all_reduce_fp32_grads)
                    optimizer.apply_gradients(zip(
                        all_reduce_fp32_grads, online_model.trainable_variables),
                        experimental_aggregate_gradients=False)

                    # Method 2
                    fp32_grads = tape.gradient(
                        loss, prediction_model.trainable_variables)
                    fp16_grads = [tf.cast(grad, 'float16')
                                  for grad in fp32_grads]
                    all_reduce_fp16_grads = tf.distribute.get_replica_context(
                    ).all_reduce(tf.distribute.ReduceOp.SUM, fp16_grads)
                    all_reduce_fp32_grads = [
                        tf.cast(grad, 'float32') for grad in all_reduce_fp16_grads]
                    # all_reduce_fp32_grads = optimizer.get_unscaled_gradients(
                    #     all_reduce_fp32_grads)
                    optimizer.apply_gradients(zip(
                        all_reduce_fp32_grads, prediction_model.trainable_variables),
                        experimental_aggregate_gradients=False)

                elif FLAGS.mixprecision == "fp32":
                    logging.info("you implement original_Fp precision")

                    # Update Encoder and Projection head weight
                    grads = tape.gradient(
                        loss, online_model.trainable_variables)
                    #grads = [ grad/ 32.0 for grad in grads]
                    optimizer.apply_gradients(
                        zip(grads, online_model.trainable_variables))

                    # Update Prediction Head model
                    grads = tape.gradient(
                        loss, prediction_model.trainable_variables)
                    #grads = [grad/ 32.0 for grad in grads]
                    optimizer.apply_gradients(
                        zip(grads, prediction_model.trainable_variables))
                
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
            alpha_base = FLAGS.alpha
            for epoch in range(FLAGS.train_epochs):
                total_loss = 0.0
                num_batches = 0
                print("Epoch", epoch, "...")
                for step, (ds_one, ds_two) in enumerate(train_ds):

                    # Update Two different Alpha Schedule for increasing Values
                    if FLAGS.alpha_schedule == "cosine_schedule":
                        logging.info("Implementation beta momentum uses Cosine Function")
                        cur_step = global_step.numpy()
                        alpha = 1 - (1 - alpha_base) * \
                            (math.cos(math.pi * cur_step / train_steps) + 1) / 2

                    elif FLAGS.alpha_schedule == "custom_schedule":
                        if epoch + 1 <= 0.7 * FLAGS.train_epochs:
                            alpha = 0.5
                            # weight_loss = 0.5
                        elif epoch + 1 <= 0.9 * FLAGS.train_epochs:
                            alpha = 0.7
                            # weight_loss = 0.7
                        else:
                            alpha = 0.9
                    else:
                        alpha = FLAGS.alpha

                    total_loss += distributed_train_step(
                        ds_one, ds_two, alpha, FLAGS.weighted_loss)
                    num_batches += 1

                    # Update weight of Target Encoder Every Step
                    #if step % 32 == 0:
                    if FLAGS.moving_average == "fixed_value":
                        beta = 0.99
                    elif FLAGS.moving_average == "schedule":
                        # This update the Beta value schedule along with Trainign steps Follow BYOL
                        logging.info(
                            "Implementation beta momentum uses Cosine Function")
                        beta_base = 0.996
                        cur_step = global_step.numpy()
                        beta = 1 - (1 - beta_base) * \
                            (cos(pi * cur_step / train_steps) + 1) / 2
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
                        logging.info('Completed: %d / %d steps',
                                     cur_step, train_steps)
                        metrics.log_and_write_metrics_to_summary(
                            all_metrics, cur_step)
                        tf.summary.scalar('learning_rate', lr_schedule(tf.cast(global_step, dtype=tf.float32)),
                                          global_step)
                        summary_writer.flush()

                epoch_loss = total_loss / num_batches
                # Wandb Configure for Visualize the Model Training
                if (epoch + 1) % 5 == 0:
                    FLAGS.train_mode = 'finetune'
                    result = perform_evaluation(online_model, val_ds, eval_steps,
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

                if (epoch + 1) % 20 == 0:
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
        save_encoder = os.path.join(
            FLAGS.model_dir, "encoder_model_latest.h5")
        save_online_model = os.path.join(
            FLAGS.model_dir, "online_model_latest.h5")
        save_target_model = os.path.join(
            FLAGS.model_dir, "target_model_latest.h5")
        online_model.encoder.save_weights(save_encoder)
        online_model.save_weights(save_online_model)
        target_model.save_weights(save_target_model)

    # Pre-Training and Finetune


if __name__ == '__main__':
    main()
