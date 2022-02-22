
import os
import random
import re
from absl import logging
import numpy as np
import tensorflow as tf
from imutils import paths
from config.absl_mock import Mock_Flag
# from byol_simclr_multi_croping_augmentation import simclr_augment_randcrop_global_views, simclr_augment_inception_style, \
#     supervised_augment_eval, simclr_augment_randcrop_global_view_image_mask, simclr_augment_inception_style_image_mask, simclr_augment_inception_style_image_mask_tf_py, simclr_augment_randcrop_global_view_image_mask_tf_py


from byol_simclr_multi_croping_augmentation import simclr_augment_randcrop_global_views, simclr_augment_inception_style, \
    supervised_augment_eval, simclr_augment_randcrop_global_view_image_mask, simclr_augment_inception_style_image_mask

# import nvidia.dali as dali
# import nvidia.dali.plugin.tf as dali_tf


AUTO = tf.data.AUTOTUNE
flag = Mock_Flag()
FLAGS = flag.FLAGS


# Experimental options
# tf.data.experimental.DistributeOptions()
options = tf.data.Options()

options.experimental_optimization.noop_elimination = True
# options.experimental_optimization.map_vectorization.enabled = True
options.experimental_optimization.map_and_batch_fusion = True
options.experimental_optimization.map_parallelization = True
options.experimental_optimization.apply_default_optimizations = True
options.experimental_deterministic = False
# options.experimental_threading.max_intra_op_parallelism = 1
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA


class imagenet_dataset_multi_machine():
    def __init__(self, img_size, train_batch, val_batch, strategy, train_path=None, train_label=None, val_path=None, val_label=None, bi_mask=False,
                 mask_path=None, subset_class_num=None, subset_percentage=None):
        '''
        args:
        img_size: Image training size
        train_batch: Distributed Batch_size for training multi-GPUs

        image_path: Directory to train data
        val_path:   Directory to validation or testing data
        subset_class_num: subset class

        '''

        self.IMG_SIZE = img_size
        self.BATCH_SIZE = train_batch
        self.val_batch = val_batch
        self.strategy = strategy
        self.seed = FLAGS.SEED
        self.bi_mask = []
        self.x_train = []
        self.x_val = []

        self.feature_size = self.IMG_SIZE / 2
        for key in list(FLAGS.Encoder_block_strides.keys()):
            self.feature_size = self.feature_size / \
                FLAGS.Encoder_block_strides[key]
        self.feature_size = int(self.feature_size)

        self.label, self.class_name = self.get_label(train_label)
        numeric_train_cls = []
        numeric_val_cls = []
        print("train_path:", train_path)
        print("val_path:", val_path)

        if train_path is None and val_path is None:
            raise ValueError(
                f'The train_path and val_path is None, please cheeek')
        elif val_path is None:
            self.get_train_path(train_path, subset_percentage)
            dataset_len = len(self.x_train)
            self.x_val = self.x_train[0:int(dataset_len * 0.2)]
            self.x_train = self.x_train[len(self.x_val) + 1:]
            for image_path in self.x_train:
                label = re.split(r"/|\|//|\\", image_path)[-2]
                # label = image_path.split("/")[-2]
                numeric_train_cls.append(self.label[label])
            for image_path in self.x_val:
                label = re.split(r"/|\|//|\\", image_path)[-2]
                numeric_val_cls.append(self.label[label])
        else:
            self.get_train_path(train_path, subset_percentage)
            self.x_val = list(paths.list_images(val_path))
            random.Random(FLAGS.SEED_data_split).shuffle(self.x_train)
            random.Random(FLAGS.SEED_data_split).shuffle(self.x_val)

            for image_path in self.x_train:
                label = re.split(r"/|\|//|\\", image_path)[-2]
                numeric_train_cls.append(self.label[label])

            val_label_map = self.get_val_label(val_label)
            numeric_val_cls = []
            for image_path in self.x_val:
                label = re.split(r"/|\|//|\\", image_path)[-1]

                label = label.split("_")[-1]
                label = int(label.split(".")[0])
                numeric_val_cls.append(val_label_map[label-1])

        if subset_class_num != None:
            x_train_sub = []
            numeric_train_cls_sub = []
            for file_path, numeric_cls in zip(self.x_train, numeric_train_cls):
                if numeric_cls < subset_class_num:
                    x_train_sub.append(file_path)
                    numeric_train_cls_sub.append(numeric_cls)
            self.x_train = x_train_sub
            numeric_train_cls = numeric_train_cls_sub

            x_val_sub = []
            numeric_val_cls_sub = []
            for file_path, numeric_cls in zip(self.x_val, numeric_val_cls):
                if numeric_cls < subset_class_num:
                    x_val_sub.append(file_path)
                    numeric_val_cls_sub.append(numeric_cls)
            self.x_val = x_val_sub
            numeric_val_cls = numeric_val_cls_sub

        if bi_mask:
            for p in self.x_train:
                self.bi_mask.append(
                    p.replace("train", mask_path).replace("JPEG", "png"))

        # Path for loading all Images
        # For training

        self.x_train_lable = tf.one_hot(numeric_train_cls, depth=len(
            self.class_name) if subset_class_num == None else subset_class_num)
        self.x_val_lable = tf.one_hot(numeric_val_cls, depth=len(
            self.class_name) if subset_class_num == None else subset_class_num)

        if bi_mask:
            self.x_train_image_mask = np.stack(
                (np.array(self.x_train), np.array(self.bi_mask)), axis=-1)
            # print(self.x_train_image_mask.shape)

    def get_train_path(self, train_path, subset_percentage):
        dir_names = os.listdir(train_path)
        for dir_name in dir_names:
            full_path = os.path.join(train_path, dir_name)
            if os.path.isdir(full_path):
                class_image_path = list(paths.list_images(full_path))
                if subset_percentage != None and len(class_image_path) > 10:
                    class_image_path = class_image_path[0:int(
                        len(class_image_path) * subset_percentage)]
                self.x_train += class_image_path

    def get_label(self, label_txt_path=None):
        class_name = []
        class_ID = []
        class_number = []
        print(label_txt_path)
        with open(label_txt_path) as file:
            for line in file.readlines():
                # n02119789 1 kit_fox
                lint_split = line.split(" ")
                class_ID.append(lint_split[0])
                class_number.append(int(lint_split[1]))
                class_name.append(lint_split[2])
            file.close()

        label = dict(zip(class_ID, class_number))
        class_name = dict(zip(class_ID, class_name))
        return label, class_name

    def get_val_label(self, label_txt_path=None):
        class_number = []
        with open(label_txt_path) as file:
            for line in file.readlines():
                class_number.append(int(line[:-1]))
                # n02119789 1 kit_fox
        return class_number

    @classmethod
    def parse_images(self, image_path):
        # Loading and reading Image
        img = tf.io.read_file(image_path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img

    @classmethod
    def parse_images_lable_pair(self, image_path, lable, IMG_SIZE):
        # Loading and reading Image
        img = tf.io.read_file(image_path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))

        return img, lable

    def prepare_mask(self, v1, v2):
        images_mask_1, lable_1, = v1
        images_mask_2, lable_2, = v2
        img1 = images_mask_1[0]
        mask1 = images_mask_1[1]
        img2 = images_mask_2[0]
        mask2 = images_mask_2[1]

        FLAGS.Encoder_block_strides
        feature_size = self.IMG_SIZE / 2
        for key in list(FLAGS.Encoder_block_strides.keys()):
            feature_size = feature_size / FLAGS.Encoder_block_strides[key]
        feature_size = int(feature_size)
        mask1 = tf.image.resize(mask1, (feature_size, feature_size))
        mask2 = tf.image.resize(mask2, (feature_size, feature_size))

        mask1 = tf.cast(mask1, dtype=tf.bool)
        mask1_obj = tf.cast(mask1, dtype=tf.float32)
        mask1_bak = tf.logical_not(mask1)
        mask1_bak = tf.cast(mask1_bak, dtype=tf.float32)

        mask2 = tf.cast(mask2, dtype=tf.bool)
        mask2_obj = tf.cast(mask2, dtype=tf.float32)
        mask2_bak = tf.logical_not(mask2)
        mask2_bak = tf.cast(mask2_bak, dtype=tf.float32)

        return (img1, mask1_obj, mask1_bak, lable_1), (img2, mask2_obj, mask2_bak, lable_2)

    @classmethod
    def parse_images_label(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.io.decode_jpeg(img, channels=3)
        label = tf.strings.split(image_path, os.path.sep)[4]
        return img, label

    @classmethod
    def parse_images_mask_lable_pair(self, image_mask_path, lable, IMG_SIZE):
        # Loading and reading Image
        # print(image_mask_path[0])
        # print(image_mask_path[1])
        image_path, mask_path = image_mask_path[0], image_mask_path[1]
        img = tf.io.read_file(image_path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))

        bi_mask = tf.io.read_file(mask_path)
        bi_mask = tf.io.decode_jpeg(bi_mask, channels=1)
        bi_mask = tf.image.resize(bi_mask, (IMG_SIZE, IMG_SIZE))

        mask = tf.cast(bi_mask, dtype=tf.bool)
        mask1_obj = tf.cast(mask, dtype=tf.float32)
        mask1_bak = tf.logical_not(mask)
        mask1_bak = tf.cast(mask1_bak, dtype=tf.float32)
        return img, mask1_obj, mask1_bak, lable

    def supervised_validation(self, input_context):
        '''This for Supervised validation training'''
        dis_tributed_batch = input_context.get_per_replica_batch_size(
            self.BATCH_SIZE)
        logging.info('Global batch size: %d', self.BATCH_SIZE)
        logging.info('Per-replica batch size: %d', dis_tributed_batch)
        logging.info('num_input_pipelines: %d',
                     input_context.num_input_pipelines)

        # options = tf.data.Options()
        # options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

        # val_ds = (tf.data.Dataset.from_tensor_slices((self.x_val, self.x_val_lable))
        #           .shuffle(self.val_batch * 100, seed=self.seed)
        #           .map(lambda x, y: (self.parse_images_lable_pair(x, y)), num_parallel_calls=AUTO)

        #           .map(lambda x, y: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE)), y),
        #                num_parallel_calls=AUTO, )
        #           .map(lambda x, y: (
        #               supervised_augment_eval(
        #                   x, FLAGS.IMG_height, FLAGS.IMG_width, FLAGS.randaug_transform, FLAGS.randaug_magnitude),
        #               y), num_parallel_calls=AUTO)

        #           )

        val_ds = tf.data.Dataset.from_tensor_slices((self.x_val, self.x_val_lable)) \
            .shuffle(self.val_batch * 100, seed=self.seed) \
            .map(lambda x, y: (self.parse_images_lable_pair(x, y, self.IMG_SIZE)), num_parallel_calls=AUTO) \
            .map(lambda x, y: (supervised_augment_eval(x, FLAGS.IMG_height, FLAGS.IMG_width, FLAGS.randaug_transform, FLAGS.randaug_magnitude),
                               y), num_parallel_calls=AUTO)

        if FLAGS.with_option:
            logging.info("You implement data loader with option")
            val_ds.with_options(options)
        else:
            logging.info("You implement data loader Without option")
            val_ds = val_ds

        val_ds = val_ds.shard(
            input_context.num_input_pipelines, input_context.input_pipeline_id)
        val_ds = val_ds.batch(dis_tributed_batch)
        # 2. modify dataset with prefetch
        val_ds = val_ds.prefetch(AUTO)

        return val_ds

    def simclr_inception_style_crop(self, input_context):
        '''
        This class property return self-supervised training data
        '''
        dis_tributed_batch = input_context.get_per_replica_batch_size(
            self.BATCH_SIZE)

        logging.info('Global batch size: %d', self.BATCH_SIZE)
        logging.info('Per-replica batch size: %d', dis_tributed_batch)
        logging.info('num_input_pipelines: %d',
                     input_context.num_input_pipelines)
        # option = tf.data.Options()
        # option.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        ds = tf.data.Dataset.from_tensor_slices((self.x_train, self.x_train_lable)) \
            .map(lambda x, y: (self.parse_images_lable_pair(x, y))) \
            .shuffle(self.BATCH_SIZE * 100, seed=self.seed) \
            .map(lambda x, y: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE)), y),).cache()

        train_ds = ds.map(lambda x, y: ((simclr_augment_inception_style(x, self.IMG_SIZE), y), (
            simclr_augment_inception_style(x, self.IMG_SIZE), y)), num_parallel_calls=AUTO)

        if FLAGS.with_option:
            logging.info("You implement data loader with option")
            train_ds.with_options(options)
        else:
            logging.info("You implement data loader Without option")
            train_ds = train_ds

        # if FLAGS.Nvidia_dali:
        #     train_ds= dali_tf.DALIDataset(
        #         train_ds,
        #         batch_size=dis_tributed_batch,
        #         device_id=input_context.input_pipeline_id
        #     )

        # else:
        train_ds = train_ds.shard(input_context.num_input_pipelines,
                                  input_context.input_pipeline_id)
        train_ds = train_ds.batch(dis_tributed_batch)
        train_ds = train_ds.prefetch(AUTO)

        return train_ds

    def simclr_random_global_crop(self, input_context):
        '''
            This class property return self-supervised training data
        '''

        dis_tributed_batch = input_context.get_per_replica_batch_size(
            self.BATCH_SIZE)
        logging.info('Global batch size: %d', self.BATCH_SIZE)
        logging.info('Per-replica batch size: %d', dis_tributed_batch)
        logging.info('num_input_pipelines: %d',
                     input_context.num_input_pipelines)

        ds = tf.data.Dataset.from_tensor_slices((self.x_train, self.x_train_lable)) \
            .map(lambda x, y: (self.parse_images_lable_pair(x, y))) \
            .shuffle(self.BATCH_SIZE * 100, seed=self.seed) \
            .map(lambda x, y: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE)), y),).cache()

        # option = tf.data.Options()
        # option.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

        train_ds = ds.map(lambda x, y: ((simclr_augment_randcrop_global_views(x, self.IMG_SIZE), y), (simclr_augment_randcrop_global_views(x, self.IMG_SIZE), y)), num_parallel_calls=AUTO)\


        if FLAGS.with_option:
            logging.info("You implement data loader with option")
            train_ds.with_options(options)
        else:
            logging.info("You implement data loader Without option")
            train_ds = train_ds

        train_ds = train_ds.shard(input_context.num_input_pipelines,
                                  input_context.input_pipeline_id)
        train_ds = train_ds.batch(dis_tributed_batch)
        train_ds = train_ds.prefetch(AUTO)

        return train_ds

    def simclr_inception_style_crop_image_mask(self, input_context):

        dis_tributed_batch = input_context.get_per_replica_batch_size(
            self.BATCH_SIZE)
        logging.info('Global batch size: %d', self.BATCH_SIZE)
        logging.info('Per-replica batch size: %d', dis_tributed_batch)
        logging.info('num_input_pipelines: %d',
                     input_context.num_input_pipelines)

        # option = tf.data.Options()
        # option.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

        ds = tf.data.Dataset.from_tensor_slices((self.x_train_image_mask, self.x_train_lable)) \
            .shuffle(self.BATCH_SIZE * 100, seed=self.seed) \
            .map(lambda x, y: (self.parse_images_mask_lable_pair(x, y, self.IMG_SIZE)), num_parallel_calls=AUTO).cache()

        # train_ds = ds.map(lambda x, y, z: ((simclr_augment_inception_style_image_mask(x, y, self.IMG_SIZE), z),
        #                                    (simclr_augment_inception_style_image_mask(x, y, self.IMG_SIZE), z)),
        #                   num_parallel_calls=AUTO) \
        #     .map(lambda x, y: self.prepare_mask(x, y), num_parallel_calls=AUTO)
        train_ds = ds.map(lambda x, y_obj, y_back, z: ((simclr_augment_inception_style_image_mask(x, y_obj, y_back, self.IMG_SIZE, self.feature_size), z),
                                                       (simclr_augment_inception_style_image_mask(x, y_obj, y_back, self.IMG_SIZE, self.feature_size), z)),
                          num_parallel_calls=AUTO)

        if FLAGS.with_option:
            logging.info("You implement data loader with option")
            train_ds.with_options(options)
        else:
            logging.info("You implement data loader Without option")

        train_ds = train_ds.shard(
            input_context.num_input_pipelines, input_context.input_pipeline_id)
        train_ds = train_ds.batch(dis_tributed_batch)
        train_ds = train_ds.prefetch(AUTO)

        return train_ds

    def simclr_random_global_crop_image_mask(self, input_context):

        dis_tributed_batch = input_context.get_per_replica_batch_size(
            self.BATCH_SIZE)
        logging.info('Global batch size: %d', self.BATCH_SIZE)
        logging.info('Per-replica batch size: %d', dis_tributed_batch)
        logging.info('num_input_pipelines: %d',
                     input_context.num_input_pipelines)

        # option = tf.data.Options()
        # option.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        ds = tf.data.Dataset.from_tensor_slices((self.x_train_image_mask, self.x_train_lable)) \
            .shuffle(self.BATCH_SIZE * 100, seed=self.seed) \
            .map(lambda x, y: (self.parse_images_mask_lable_pair(x, y, self.IMG_SIZE)), num_parallel_calls=AUTO).cache()

        # train_ds = ds.map(lambda x, y, z: ((simclr_augment_randcrop_global_view_image_mask(x, y, self.IMG_SIZE), z),
        #                                    (simclr_augment_randcrop_global_view_image_mask(x, y, self.IMG_SIZE), z)),
        #                   num_parallel_calls=AUTO) \
        #     .map(lambda x, y: self.prepare_mask(x, y), num_parallel_calls=AUTO)
        train_ds = ds.map(lambda x, y_obj, y_back, z: ((simclr_augment_randcrop_global_view_image_mask(x, y_obj, y_back, self.IMG_SIZE, self.feature_size), z),
                                                       (simclr_augment_randcrop_global_view_image_mask(x, y_obj, y_back, self.IMG_SIZE, self.feature_size), z)),
                          num_parallel_calls=AUTO)

        if FLAGS.with_option:
            logging.info("You implement data loader with option")
            train_ds.with_options(options)
        else:
            logging.info("You implement data loader Without option")

        train_ds = train_ds.shard(input_context.num_input_pipelines,
                                  input_context.input_pipeline_id)
        train_ds = train_ds.batch(dis_tributed_batch)
        train_ds = train_ds.prefetch(AUTO)

        return train_ds

    def get_data_size(self):
        return len(self.x_train), len(self.x_val)


if __name__ == "__main__":
    # strategy = tf.distribute.MirroredStrategy()
    # train_dataset = imagenet_dataset_single_machine(img_size=256, train_batch=32, val_batch=32,
    #                                                 strategy=strategy, train_path=None, val_path=None, bi_mask=True)

    # train_ds = train_dataset.simclr_random_global_crop_image_mask()

    # val_ds = train_dataset.supervised_validation()

    # num_train_examples = len(train_ds)
    # # num_eval_examples = len(val_ds)

    # print("num_train_examples : ", num_train_examples)
    # print("num_eval_examples : ", num_eval_examples)
    print("test")
