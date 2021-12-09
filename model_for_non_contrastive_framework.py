# coding=utf-8
# Copyright 2020 The SimCLR Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific simclr governing permissions and
# limitations under the License.
# ==============================================================================
"""Model specification for SimCLR."""

import math
from absl import flags


import lars_optimizer
import resnet
from Model_resnet_harry import resnet as resnet_modify
import tensorflow as tf
from learning_rate_optimizer import get_optimizer
from tensorflow.keras import mixed_precision
from visualize import Visualize

#FLAGS = flags.FLAGS

from config.absl_mock import Mock_Flag
flag = Mock_Flag()
FLAGS = flag.FLAGS

def build_optimizer(lr_schedule):
    '''
    Args
    lr_schedule: learning values.

    Return:
    'original', 'optimizer_weight_decay','optimizer_GD','optimizer_W_GD' 
    optimizer.
    '''
    if FLAGS.optimizer_type == "original":
        Optimizer_type = FLAGS.optimizer
        optimizers = get_optimizer(lr_schedule, Optimizer_type)
        optimizer = optimizers.original_optimizer(FLAGS)
    elif FLAGS.optimizer_type == "optimizer_weight_decay":
        Optimizer_type = FLAGS.optimizer
        optimizers = get_optimizer(lr_schedule, Optimizer_type)
        optimizer = optimizers.optimizer_weight_decay(FLAGS)

    elif FLAGS.optimizer_type == "optimizer_GD":
        Optimizer_type = FLAGS.optimizer
        optimizers = get_optimizer(lr_schedule, Optimizer_type)
        optimizer = optimizers.optimizer_gradient_centralization(FLAGS)

    elif FLAGS.optimizer_type == "optimizer_W_GD":
        Optimizer_type = FLAGS.optimizer
        optimizers = get_optimizer(lr_schedule, Optimizer_type)
        optimizer = optimizers.optimizer_weight_decay_gradient_centralization(
            FLAGS)
    else:
        raise ValueError(" FLAGS.Optimizer type is invalid please check again")
    #optimizer_mix_percision = mixed_precision.LossScaleOptimizer(optimizer)

    return optimizer


def build_optimizer_multi_machine(lr_schedule):
    '''
    Args
    lr_schedule: learning values.

    Return:
    The mix_percision optimizer.'optimizer_weight_decay','optimizer_GD','optimizer_W_GD' 
    '''

    if FLAGS.optimizer_type == "original":
        Optimizer_type = FLAGS.optimizer
        optimizers = get_optimizer(lr_schedule, Optimizer_type)
        optimizer = optimizers.original_optimizer(FLAGS)
        optimizer_mix_percision = mixed_precision.LossScaleOptimizer(optimizer)
    elif FLAGS.optimizer_type == "optimizer_weight_decay":
        Optimizer_type = FLAGS.optimizer
        optimizers = get_optimizer(lr_schedule, Optimizer_type)
        optimizer = optimizers.optimizer_weight_decay(FLAGS)
        optimizer_mix_percision = mixed_precision.LossScaleOptimizer(optimizer)

    elif FLAGS.optimizer_type == "optimizer_GD":
        Optimizer_type = FLAGS.optimizer
        optimizers = get_optimizer(lr_schedule, Optimizer_type)
        optimizer = optimizers.optimizer_gradient_centralization(FLAGS)
        optimizer_mix_percision = mixed_precision.LossScaleOptimizer(optimizer)

    elif FLAGS.optimizer_type == "optimizer_W_GD":
        Optimizer_type = FLAGS.optimizer
        optimizers = get_optimizer(lr_schedule, Optimizer_type)
        optimizer = optimizers.optimizer_weight_decay_gradient_centralization(
            FLAGS)
        optimizer_mix_percision = mixed_precision.LossScaleOptimizer(optimizer)
    else:
        raise ValueError(" FLAGS.Optimizer type is invalid please check again")
    #optimizer_mix_percision = mixed_precision.LossScaleOptimizer(optimizer)

    return optimizer_mix_percision


def add_weight_decay(model, adjust_per_optimizer=True):
    """Compute weight decay from flags."""
    if adjust_per_optimizer and 'lars' in FLAGS.optimizer:
        # Weight decay are taking care of by optimizer for these cases.
        # Except for supervised head, which will be added here.
        l2_losses = [
            tf.nn.l2_loss(v)
            for v in model.trainable_variables
            if 'head_supervised' in v.name and 'bias' not in v.name
        ]
        if l2_losses:
            return FLAGS.weight_decay * tf.add_n(l2_losses)
        else:
            return 0

    # TODO(srbs): Think of a way to avoid name-based filtering here.
    l2_losses = [
        tf.nn.l2_loss(v)
        for v in model.trainable_weights
        if 'batch_normalization' not in v.name
    ]

    loss = FLAGS.weight_decay * tf.add_n(l2_losses)

    return loss


# Linear Layers tf.keras.layer.Dense
class modify_LinearLayer(tf.keras.layers.Layer):

    def __init__(self,
                 num_classes,
                 up_scale=4096,
                 non_contrastive=False,
                 use_bias=True,
                 use_bn=False,
                 name='linear_layer',
                 **kwargs):
        # Note: use_bias is ignored for the dense layer when use_bn=True.
        # However, it is still used for batch norm.
        super(modify_LinearLayer, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.up_scale = up_scale
        self.use_bias = use_bias
        self.use_bn = use_bn
        self._name = name
        self.non_contrastive = non_contrastive
        if self.use_bn:
            self.bn_relu = resnet.BatchNormRelu(relu=False, center=use_bias)
            #self.bn_relu= tf.keras.layers.BatchNormalization()

    def build(self, input_shape):
        # TODO(srbs): Add a new SquareDense layer.
        if callable(self.num_classes):
            num_classes = self.num_classes(input_shape)

        else:
            num_classes = self.num_classes

        self.dense_upscale = tf.keras.layers.Dense(
            self.up_scale,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            use_bias=self.use_bias and not self.use_bn, )

        self.dense = tf.keras.layers.Dense(
            num_classes,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            use_bias=self.use_bias and not self.use_bn)

        super(modify_LinearLayer, self).build(input_shape)

    def call(self, inputs, training):
        assert inputs.shape.ndims == 2, inputs.shape
        if self.non_contrastive:
            inputs = self.dense_upscale(inputs)
            # print(inputs.shape)
            #inputs = self.dense(inputs)
            if self.use_bn:
                inputs = self.bn_relu(inputs, training=training)
        else:
            inputs = self.dense(inputs)
            # print(inputs.shape)
            if self.use_bn:
                inputs = self.bn_relu(inputs, training=training)
        return inputs

# Projection Head add  Batchnorm layer


class ProjectionHead(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        out_dim = FLAGS.proj_out_dim
        self.linear_layers = []
        if FLAGS.proj_head_mode == 'none':
            pass  # directly use the output hiddens as hiddens
        elif FLAGS.proj_head_mode == 'linear':
            self.linear_layers = [
                modify_LinearLayer(
                    num_classes=out_dim,  use_bias=False, use_bn=True, name='l_0')
            ]
        elif FLAGS.proj_head_mode == 'nonlinear':
            if FLAGS.num_proj_layers > 2:
                for j in range(FLAGS.num_proj_layers):
                    if j == 0:
                        self.linear_layers.append(
                            modify_LinearLayer(
                                num_classes=lambda input_shape: int(
                                    input_shape[-1]),
                                up_scale=FLAGS.up_scale, non_contrastive=FLAGS.non_contrastive,
                                use_bias=True,
                                use_bn=True,
                                name='nl_%d' % j))

                    elif j != FLAGS.num_proj_layers - 1:
                        # for the middle layers, use bias and relu for the output.
                        if FLAGS.reduce_linear_dimention:
                            print("You Implement reduction")
                            self.linear_layers.append(
                                modify_LinearLayer(
                                    num_classes=lambda input_shape: int(
                                        input_shape[-1]/2),
                                    up_scale=FLAGS.up_scale, non_contrastive=False,
                                    use_bias=True,
                                    use_bn=True,
                                    name='nl_%d' % j))
                        else:
                            self.linear_layers.append(
                                modify_LinearLayer(
                                    num_classes=lambda input_shape: int(
                                        input_shape[-1]),
                                    up_scale=FLAGS.up_scale, non_contrastive=False,
                                    use_bias=True,
                                    use_bn=True,
                                    name='nl_%d' % j))

                    else:
                        # for the final layer, neither bias nor relu is used.
                        self.linear_layers.append(
                            modify_LinearLayer(
                                num_classes=FLAGS.proj_out_dim,
                                use_bias=False,
                                use_bn=True,
                                name='nl_%d' % j))

            else:
                for j in range(FLAGS.num_proj_layers):
                    if j != FLAGS.num_proj_layers - 1:
                        # for the middle layers, use bias and relu for the output.
                        self.linear_layers.append(
                            modify_LinearLayer(
                                num_classes=lambda input_shape: int(
                                    input_shape[-1]),
                                up_scale=FLAGS.up_scale, non_contrastive=FLAGS.non_contrastive,
                                use_bias=True,
                                use_bn=True,
                                name='nl_%d' % j))
                    else:
                        # for the final layer, neither bias nor relu is used.
                        self.linear_layers.append(
                            modify_LinearLayer(
                                num_classes=FLAGS.proj_out_dim,
                                up_scale=FLAGS.up_scale, non_contrastive=False,
                                use_bias=False,
                                use_bn=True,
                                name='nl_%d' % j))
        else:
            raise ValueError('Unknown head projection mode {}'.format(
                FLAGS.proj_head_mode))

        super(ProjectionHead, self).__init__(**kwargs)

    def call(self, inputs, training):
        if FLAGS.proj_head_mode == 'none':
            return inputs  # directly use the output hiddens as hiddens
        hiddens_list = [tf.identity(inputs, 'proj_head_input')]
        if FLAGS.proj_head_mode == 'linear':
            assert len(self.linear_layers) == 1, len(self.linear_layers)
            return hiddens_list.append(self.linear_layers[0](hiddens_list[-1],
                                                             training))

        elif FLAGS.proj_head_mode == 'nonlinear':
            for j in range(FLAGS.num_proj_layers):
                hiddens = self.linear_layers[j](hiddens_list[-1], training)
                if j != FLAGS.num_proj_layers - 1:
                    # for the middle layers, use bias and relu for the output.
                    hiddens = tf.nn.relu(hiddens)
                hiddens_list.append(hiddens)

        else:
            raise ValueError('Unknown head projection mode {}'.format(
                FLAGS.proj_head_mode))

        # The first element is the output of the projection head.
        # The second element is the input of the finetune head.
        proj_head_output = tf.identity(hiddens_list[-1], 'proj_head_output')

        return proj_head_output, hiddens_list[FLAGS.ft_proj_selector]


class LinearLayer(tf.keras.layers.Layer):

    def __init__(self,
                 num_classes,
                 use_bias=True,
                 use_bn=False,
                 name='linear_layer',
                 **kwargs):
        # Note: use_bias is ignored for the dense layer when use_bn=True.
        # However, it is still used for batch norm.
        super(LinearLayer, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.use_bias = use_bias
        self.use_bn = use_bn
        self._name = name
        if self.use_bn:
            self.bn_relu = resnet.BatchNormRelu(relu=False, center=use_bias)

    def build(self, input_shape):
        # TODO(srbs): Add a new SquareDense layer.
        if callable(self.num_classes):
            num_classes = self.num_classes(input_shape)
        else:
            num_classes = self.num_classes
        self.dense = tf.keras.layers.Dense(
            num_classes,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            use_bias=self.use_bias and not self.use_bn)

        super(LinearLayer, self).build(input_shape)

    def call(self, inputs, training):
        assert inputs.shape.ndims == 2, inputs.shape
        inputs = self.dense(inputs)
        if self.use_bn:
            inputs = self.bn_relu(inputs, training=training)
        return inputs


class ProjectionHead_original(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        out_dim = FLAGS.proj_out_dim
        self.linear_layers = []
        if FLAGS.proj_head_mode == 'none':
            pass  # directly use the output hiddens as hiddens
        elif FLAGS.proj_head_mode == 'linear':

            self.linear_layers = [
                LinearLayer(
                    num_classes=out_dim, use_bias=False, use_bn=True, name='l_0')
            ]
        elif FLAGS.proj_head_mode == 'nonlinear':

            for j in range(FLAGS.num_proj_layers):
                if j != FLAGS.num_proj_layers - 1:
                    # for the middle layers, use bias and relu for the output.
                    self.linear_layers.append(
                        LinearLayer(
                            num_classes=lambda input_shape: int(
                                input_shape[-1]),
                            use_bias=True,
                            use_bn=True,
                            name='nl_%d' % j))
                else:
                    # for the final layer, neither bias nor relu is used.
                    self.linear_layers.append(
                        LinearLayer(
                            num_classes=FLAGS.proj_out_dim,
                            use_bias=False,
                            use_bn=True,
                            name='nl_%d' % j))
        else:
            raise ValueError('Unknown head projection mode {}'.format(
                FLAGS.proj_head_mode))

        super(ProjectionHead, self).__init__(**kwargs)

    def call(self, inputs, training):
        if FLAGS.proj_head_mode == 'none':
            return inputs  # directly use the output hiddens as hiddens
        hiddens_list = [tf.identity(inputs, 'proj_head_input')]

        if FLAGS.proj_head_mode == 'linear':
            assert len(self.linear_layers) == 1, len(self.linear_layers)
            return hiddens_list.append(self.linear_layers[0](hiddens_list[-1],
                                                             training))
        elif FLAGS.proj_head_mode == 'nonlinear':
            for j in range(FLAGS.num_proj_layers):
                hiddens = self.linear_layers[j](hiddens_list[-1], training)
                if j != FLAGS.num_proj_layers - 1:
                    # for the middle layers, use bias and relu for the output.
                    hiddens = tf.nn.relu(hiddens)
                hiddens_list.append(hiddens)

        else:
            raise ValueError('Unknown head projection mode {}'.format(
                FLAGS.proj_head_mode))

        # The first element is the output of the projection head.
        # The second element is the input of the finetune head.
        proj_head_output = tf.identity(hiddens_list[-1], 'proj_head_output')

        return proj_head_output, hiddens_list[FLAGS.ft_proj_selector]


class SupervisedHead(tf.keras.layers.Layer):

    def __init__(self, num_classes, name='head_supervised', **kwargs):
        super(SupervisedHead, self).__init__(name=name, **kwargs)
        self.linear_layer = modify_LinearLayer(num_classes)

    def call(self, inputs, training):
        inputs = self.linear_layer(inputs, training)
        inputs = tf.identity(inputs, name='logits_sup')
        return inputs

# Projection Head add  Batchnorm layer

# Also Need input (Batch_size, Dim)


class PredictionHead(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        out_dim = FLAGS.prediction_out_dim
        self.linear_layers = []
        if FLAGS.proj_head_mode == 'none':
            pass  # directly use the output hiddens as hiddens
        elif FLAGS.proj_head_mode == 'linear':
            self.linear_layers = [
                modify_LinearLayer(
                    num_classes=out_dim,  use_bias=False, use_bn=True, name='l_0')
            ]
        elif FLAGS.proj_head_mode == 'nonlinear':
            if FLAGS.num_proj_layers > 2:
                for j in range(FLAGS.num_proj_layers):
                    if j == 0:
                        self.linear_layers.append(
                            modify_LinearLayer(
                                num_classes=lambda input_shape: int(
                                    input_shape[-1]),
                                up_scale=FLAGS.up_scale, non_contrastive=FLAGS.non_contrastive,
                                use_bias=True,
                                use_bn=True,
                                name='nl_%d' % j))

                    elif j != FLAGS.num_proj_layers - 1:
                        # for the middle layers, use bias and relu for the output.
                        if FLAGS.reduce_linear_dimention:
                            print("Implement reduce dimention")
                            self.linear_layers.append(
                                modify_LinearLayer(
                                    num_classes=lambda input_shape: int(
                                        input_shape[-1]/2),
                                    up_scale=FLAGS.up_scale, non_contrastive=False,
                                    use_bias=True,
                                    use_bn=True,
                                    name='nl_%d' % j))
                        else:
                            self.linear_layers.append(
                                modify_LinearLayer(
                                    num_classes=lambda input_shape: int(
                                        input_shape[-1]),
                                    up_scale=FLAGS.up_scale, non_contrastive=False,
                                    use_bias=True,
                                    use_bn=True,
                                    name='nl_%d' % j))

                    else:
                        # for the final layer, neither bias nor relu is used.
                        self.linear_layers.append(
                            modify_LinearLayer(
                                num_classes=FLAGS.proj_out_dim,
                                use_bias=False,
                                use_bn=True,
                                name='nl_%d' % j))

            else:
                for j in range(FLAGS.num_proj_layers):
                    if j != FLAGS.num_proj_layers - 1:
                        # for the middle layers, use bias and relu for the output.
                        self.linear_layers.append(
                            modify_LinearLayer(
                                num_classes=lambda input_shape: int(
                                    input_shape[-1]),
                                up_scale=FLAGS.up_scale, non_contrastive=FLAGS.non_contrastive,
                                use_bias=True,
                                use_bn=True,
                                name='nl_%d' % j))
                    else:
                        # for the final layer, neither bias nor relu is used.
                        self.linear_layers.append(
                            modify_LinearLayer(
                                num_classes=FLAGS.proj_out_dim,
                                up_scale=FLAGS.up_scale, non_contrastive=False,
                                use_bias=False,
                                use_bn=True,
                                name='nl_%d' % j))
        else:
            raise ValueError('Unknown head projection mode {}'.format(
                FLAGS.proj_head_mode))

        super(PredictionHead, self).__init__(**kwargs)

    def call(self, inputs, training):
        if FLAGS.proj_head_mode == 'none':
            return inputs  # directly use the output hiddens as hiddens
        hiddens_list = [tf.identity(inputs, 'proj_head_input')]
        if FLAGS.proj_head_mode == 'linear':
            assert len(self.linear_layers) == 1, len(self.linear_layers)
            return hiddens_list.append(self.linear_layers[0](hiddens_list[-1],
                                                             training))

        elif FLAGS.proj_head_mode == 'nonlinear':
            for j in range(FLAGS.num_proj_layers):
                hiddens = self.linear_layers[j](hiddens_list[-1], training)
                if j != FLAGS.num_proj_layers - 1:
                    # for the middle layers, use bias and relu for the output.
                    hiddens = tf.nn.relu(hiddens)
                hiddens_list.append(hiddens)

        else:
            raise ValueError('Unknown head projection mode {}'.format(
                FLAGS.proj_head_mode))

        # The first element is the output of the projection head.
        # The second element is the input of the finetune head.
        proj_head_output = tf.identity(hiddens_list[-1], 'proj_head_output')
        return proj_head_output


class PredictionHead_original(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        out_dim = FLAGS.prediction_out_dim
        self.linear_layers = []
        if FLAGS.proj_head_mode == 'none':
            pass  # directly use the output hiddens as hiddens
        elif FLAGS.proj_head_mode == 'linear':

            self.linear_layers = [
                LinearLayer(
                    num_classes=out_dim, use_bias=False, use_bn=True, name='l_0')
            ]
        elif FLAGS.proj_head_mode == 'nonlinear':

            for j in range(FLAGS.num_proj_layers):
                if j != FLAGS.num_proj_layers - 1:
                    # for the middle layers, use bias and relu for the output.
                    self.linear_layers.append(
                        LinearLayer(
                            num_classes=lambda input_shape: int(
                                input_shape[-1]),
                            use_bias=True,
                            use_bn=True,
                            name='nl_%d' % j))
                else:
                    # for the final layer, neither bias nor relu is used.
                    self.linear_layers.append(
                        LinearLayer(
                            num_classes=FLAGS.proj_out_dim,
                            use_bias=False,
                            use_bn=True,
                            name='nl_%d' % j))
        else:
            raise ValueError('Unknown head projection mode {}'.format(
                FLAGS.proj_head_mode))

        super(PredictionHead, self).__init__(**kwargs)

    def call(self, inputs, training):
        if FLAGS.proj_head_mode == 'none':
            return inputs  # directly use the output hiddens as hiddens
        hiddens_list = [tf.identity(inputs, 'proj_head_input')]

        if FLAGS.proj_head_mode == 'linear':
            assert len(self.linear_layers) == 1, len(self.linear_layers)
            return hiddens_list.append(self.linear_layers[0](hiddens_list[-1],
                                                             training))
        elif FLAGS.proj_head_mode == 'nonlinear':
            for j in range(FLAGS.num_proj_layers):
                hiddens = self.linear_layers[j](hiddens_list[-1], training)
                if j != FLAGS.num_proj_layers - 1:
                    # for the middle layers, use bias and relu for the output.
                    hiddens = tf.nn.relu(hiddens)
                hiddens_list.append(hiddens)

        else:
            raise ValueError('Unknown head projection mode {}'.format(
                FLAGS.proj_head_mode))

        # The first element is the output of the projection head.
        # The second element is the input of the finetune head.
        proj_head_output = tf.identity(hiddens_list[-1], 'proj_head_output')

        return proj_head_output


"""# Indexer"""


class Indexer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(Indexer, self).__init__(**kwargs)

    def call(self, input):
        feature_map = input[0]
        mask = input[1]
        if feature_map.shape[1] != mask.shape[1] and feature_map.shape[2] != mask.shape[2]:
            mask = tf.image.resize(
                mask, (feature_map.shape[1], feature_map.shape[2]))
        mask = tf.cast(mask, dtype=tf.bool)
        mask = tf.cast(mask, dtype=feature_map.dtype)
        obj = tf.multiply(feature_map, mask)
        mask = tf.cast(mask, dtype=tf.bool)
        mask = tf.logical_not(mask)
        mask = tf.cast(mask, dtype=feature_map.dtype)
        back = tf.multiply(feature_map, mask)
        return obj, back


class prediction_head_model(tf.keras.models.Model):
    def __init__(self, **kwargs):

        super(prediction_head_model, self).__init__(**kwargs)
        # prediction head
        self._prediction_head = PredictionHead()

    def __call__(self, inputs, training):
        prediction_head_outputs = self._prediction_head(inputs, training)
        return prediction_head_outputs


class online_model(tf.keras.models.Model):
    """Resnet model with projection or supervised layer."""

    def __init__(self, num_classes, **kwargs):

        super(online_model, self).__init__(**kwargs)
        # Encoder
        self.resnet_model = resnet.resnet(
            resnet_depth=FLAGS.resnet_depth,
            width_multiplier=FLAGS.width_multiplier,
            cifar_stem=FLAGS.image_size <= 32)
        # Projcetion head
        self._projection_head = ProjectionHead()

        # Supervised classficiation head
        if FLAGS.train_mode == 'finetune' or FLAGS.lineareval_while_pretraining:
            self.supervised_head = SupervisedHead(num_classes)

    def __call__(self, inputs, training):

        features = inputs

        if training and FLAGS.train_mode == 'pretrain':
            if FLAGS.fine_tune_after_block > -1:
                raise ValueError('Does not support layer freezing during pretraining,'
                                 'should set fine_tune_after_block<=-1 for safety.')

        if inputs.shape[3] is None:
            raise ValueError('The input channels dimension must be statically known '
                             f'(got input shape {inputs.shape})')

        # # Base network forward pass.
        hiddens = self.resnet_model(features, training=training)
        #print("Output from ResNet Model", hiddens.shape)
        # Add heads.
        projection_head_outputs, supervised_head_inputs, = self._projection_head(
            hiddens, training)

        #print("output from  Online projection Head",projection_head_outputs.shape )

        if FLAGS.train_mode == 'finetune':
            supervised_head_outputs = self.supervised_head(supervised_head_inputs,
                                                           training)
            return None, supervised_head_outputs

        elif FLAGS.train_mode == 'pretrain' and FLAGS.lineareval_while_pretraining:
            # When performing pretraining and linear evaluation together we do not
            # want information from linear eval flowing back into pretraining network
            # so we put a stop_gradient.
            supervised_head_outputs = self.supervised_head(
                tf.stop_gradient(supervised_head_inputs), training)
            #print("Supervised Head Output Dim", supervised_head_outputs.shape)
            return projection_head_outputs, supervised_head_outputs

        else:
            return projection_head_outputs, None

# Consideration take Supervised evaluate From the Target model


class target_model(tf.keras.models.Model):
    """Resnet model with projection or supervised layer."""

    def __init__(self, num_classes, **kwargs):

        super(target_model, self).__init__(**kwargs)
        # Encoder
        self.resnet_model = resnet.resnet(
            resnet_depth=FLAGS.resnet_depth,
            width_multiplier=FLAGS.width_multiplier,
            cifar_stem=FLAGS.image_size <= 32)
        # Projcetion head
        self._projection_head = ProjectionHead()

        # Supervised classficiation head
        if FLAGS.train_mode == 'finetune' or FLAGS.lineareval_while_pretraining:
            self.supervised_head = SupervisedHead(num_classes)

    def __call__(self, inputs, training):

        features = inputs

        if training and FLAGS.train_mode == 'pretrain':
            if FLAGS.fine_tune_after_block > -1:
                raise ValueError('Does not support layer freezing during pretraining,'
                                 'should set fine_tune_after_block<=-1 for safety.')

        if inputs.shape[3] is None:
            raise ValueError('The input channels dimension must be statically known '
                             f'(got input shape {inputs.shape})')

        # # Base network forward pass.
        hiddens = self.resnet_model(features, training=training)

        # Add heads.
        projection_head_outputs, supervised_head_inputs = self._projection_head(
            hiddens, training)

        if FLAGS.train_mode == 'finetune':
            supervised_head_outputs = self.supervised_head(supervised_head_inputs,
                                                           training)
            return None, supervised_head_outputs

        elif FLAGS.train_mode == 'pretrain' and FLAGS.lineareval_while_pretraining:
            # When performing pretraining and linear evaluation together we do not
            # want information from linear eval flowing back into pretraining network
            # so we put a stop_gradient.
            supervised_head_outputs = self.supervised_head(
                tf.stop_gradient(supervised_head_inputs), training)

            return projection_head_outputs, supervised_head_outputs

        else:
            return projection_head_outputs, None


class Downsample_Layear(tf.keras.layers.Layer):
    def __init__(self,mod,**kwargs):
        super(Downsample_Layear, self).__init__(**kwargs)
        self.mod = mod
        self.globalaveragepooling = tf.keras.layers.GlobalAveragePooling2D()
        self.maxpooling = tf.keras.layers.MaxPooling2D(pool_size = (2,2), strides = FLAGS.downsample_magnification)
        self.avergepooling = tf.keras.layers.AveragePooling2D(pool_size = (2,2), strides = FLAGS.downsample_magnification)
        self.flatten = tf.keras.layers.Flatten()

    def call(self,x,k=1):
        # if k == 1:
        #     x = self.globalaveragepooling(x)
        # el
        if self.mod == "maxpooling":
            x = self.maxpooling(x)
            x = self.flatten(x)
        elif self.mod == "averagepooling":
            x = self.avergepooling(x)
            x = self.flatten(x)
        else:
            if k != 1:
                x = tf.nn.space_to_depth(x, k)
            x = self.globalaveragepooling(x)
        return x


class Binary_online_model(tf.keras.models.Model):
    """Resnet model with projection or supervised layer."""

    def __init__(self, num_classes, Backbone="Resnet", Upsample = True,Downsample = "maxpooling", **kwargs):
        super(Binary_online_model, self).__init__(**kwargs)
        self.Upsample = FLAGS.feature_upsample
        self.magnification = 1
        # Encoder
        if Backbone == "Resnet":
            self.encoder = resnet_modify(resnet_depth=FLAGS.resnet_depth,
                                         width_multiplier=FLAGS.width_multiplier)
        else:
            raise ValueError(f"Didn't have this {Backbone} model")

        # Projcetion head
        self.projection_head = ProjectionHead()
        if FLAGS.Middle_layer_output == 0:
            self.full_image_projection_head = self.projection_head
        else:
            self.full_image_projection_head = ProjectionHead()

        self.indexer = Indexer()

        self.maxpooling = tf.keras.layers.MaxPooling2D(4, 4)

        self.flatten = tf.keras.layers.Flatten()

        self.globalaveragepooling = tf.keras.layers.GlobalAveragePooling2D()
        # Supervised classficiation head
        if FLAGS.train_mode == 'finetune' or FLAGS.lineareval_while_pretraining:
            self.supervised_head = SupervisedHead(num_classes)

        self.downsample_layear=Downsample_Layear(Downsample)

    def call(self, inputs, training):

        if FLAGS.train_mode == 'pretrain':
            mask = inputs[1]
            inputs = inputs[0]

        if training and FLAGS.train_mode == 'pretrain':
            if FLAGS.fine_tune_after_block > -1:
                raise ValueError('Does not support layer freezing during pretraining,'
                                 'should set fine_tune_after_block<=-1 for safety.')
        if inputs.shape[3] is None:
            raise ValueError('The input channels dimension must be statically known '
                             f'(got input shape {inputs.shape})')

        # Base network forward pass
        org_feature_map = None
        if FLAGS.Middle_layer_output == 0:
            feature_map = self.encoder(inputs, training=training)
        else:
            org_feature_map, feature_map = self.encoder(inputs, training=training)
            print("Middle output size : ",feature_map.shape)
            # return feature_map


        if self.Upsample:
            # Pixel shuffle
            self.magnification = mask.shape[1]/feature_map.shape[1]
            feature_map_upsample = tf.nn.depth_to_space(
                feature_map, self.magnification)  # PixelShuffle
        else:
            self.magnification = FLAGS.downsample_magnification
            feature_map_upsample = feature_map
            if FLAGS.visualize:
                return feature_map

        #print("feature_map_upsample", feature_map_upsample.shape)

        # Add heads
        if FLAGS.train_mode == 'pretrain':
            # object and background indexer
            obj, back = self.indexer([feature_map_upsample, mask])
            obj, _ = self.projection_head(self.downsample_layear(obj,self.magnification)
                                          , training=training)
            back, _ = self.projection_head(self.downsample_layear(back,self.magnification)
                                          , training=training)

        if org_feature_map != None and FLAGS.non_contrast_binary_loss == "sum_symetrize_l2_loss_object_backg_add_original":
            org_feature_map = self.downsample_layear(org_feature_map,self.magnification)
            print(org_feature_map.shape)
            projection_head_outputs, supervised_head_inputs = self.full_image_projection_head(org_feature_map, training=training)
        else:
            projection_head_outputs, supervised_head_inputs = self.projection_head(self.downsample_layear(feature_map_upsample,self.magnification)
                                                                               , training=training)

        if FLAGS.train_mode == 'finetune':
            print(supervised_head_inputs)
            supervised_head_outputs = self.supervised_head(
                supervised_head_inputs, training)
            return None, None, None, supervised_head_outputs

        elif FLAGS.train_mode == 'pretrain' and FLAGS.lineareval_while_pretraining:
            # When performing pretraining and linear evaluation together we do not
            # want information from linear eval flowing back into pretraining network
            # so we put a stop_gradient.
            supervised_head_outputs = self.supervised_head(
                tf.stop_gradient(supervised_head_inputs), training)

            return obj, back, projection_head_outputs, supervised_head_outputs #,feature_map_upsample

        else:
            return obj, back, projection_head_outputs, None

        return obj, back, feature_map_upsample
        # return feature_map_upsample

# Consideration take Supervised evaluate From the Target model


class Binary_target_model(tf.keras.models.Model):
    """Resnet model with projection or supervised layer."""

    def __init__(self, num_classes, Backbone="Resnet", Upsample = True,Downsample = "maxpooling",  **kwargs):
        super(Binary_target_model, self).__init__(**kwargs)
        self.Upsample = FLAGS.feature_upsample
        self.magnification = 1
        # Encoder
        if Backbone == "Resnet":
            self.encoder = resnet_modify(resnet_depth=FLAGS.resnet_depth,
                                         width_multiplier=FLAGS.width_multiplier)
        else:
            raise ValueError(f"Didn't have this {Backbone} model")

        # Projcetion head
        self.projection_head = ProjectionHead()
        if FLAGS.Middle_layer_output == 0:
            self.full_image_projection_head = self.projection_head
        else:
            self.full_image_projection_head = ProjectionHead()

        self.indexer = Indexer()

        self.maxpooling = tf.keras.layers.MaxPooling2D(4, 4)

        self.flatten = tf.keras.layers.Flatten()

        self.globalaveragepooling = tf.keras.layers.GlobalAveragePooling2D()
        # Supervised classficiation head
        if FLAGS.train_mode == 'finetune' or FLAGS.lineareval_while_pretraining:
            self.supervised_head = SupervisedHead(num_classes)

        self.downsample_layear = Downsample_Layear(Downsample)

    def call(self, inputs, training):

        if FLAGS.train_mode == 'pretrain':
            mask = inputs[1]
            inputs = inputs[0]

        if training and FLAGS.train_mode == 'pretrain':
            if FLAGS.fine_tune_after_block > -1:
                raise ValueError('Does not support layer freezing during pretraining,'
                                 'should set fine_tune_after_block<=-1 for safety.')
        if inputs.shape[3] is None:
            raise ValueError('The input channels dimension must be statically known '
                             f'(got input shape {inputs.shape})')

        # Base network forward pass
        org_feature_map = None
        if FLAGS.Middle_layer_output == 0:
            feature_map = self.encoder(inputs, training=training)
        else:
            org_feature_map, feature_map = self.encoder(inputs, training=training)
            print("Middle output size : ",feature_map.shape)
            if FLAGS.visualize:
                return feature_map

        # Pixel shuffle
        if self.Upsample:
            # Pixel shuffle
            self.magnification = mask.shape[1]/feature_map.shape[1]
            feature_map_upsample = tf.nn.depth_to_space(
                feature_map, self.magnification)  # PixelShuffle
        else:
            self.magnification = FLAGS.downsample_magnification
            feature_map_upsample = feature_map
        #print("feature_map_upsample", feature_map_upsample.shape)

        # Add heads
        if FLAGS.train_mode == 'pretrain':
            # object and background indexer
            obj, back = self.indexer([feature_map_upsample, mask])
            obj, _ = self.projection_head(self.downsample_layear(obj,self.magnification), training=training)
            back, _ = self.projection_head(self.downsample_layear(back,self.magnification), training=training)
            # if FLAGS.visualize:
            #     self.visualize.plot_feature_map("obj",obj)
            #     self.visualize.plot_feature_map("back",obj)

        if org_feature_map != None and FLAGS.non_contrast_binary_loss == "sum_symetrize_l2_loss_object_backg_add_original":
            org_feature_map = self.downsample_layear(org_feature_map,self.magnification)
            print(org_feature_map.shape)
            projection_head_outputs, supervised_head_inputs = self.full_image_projection_head(org_feature_map, training=training)
        else:
            projection_head_outputs, supervised_head_inputs = self.projection_head(self.downsample_layear(feature_map_upsample,self.magnification)
                                                                               , training=training)
        if FLAGS.train_mode == 'finetune':
            supervised_head_outputs = self.supervised_head(
                supervised_head_inputs, training)
            return None, None, None, supervised_head_outputs

        elif FLAGS.train_mode == 'pretrain' and FLAGS.lineareval_while_pretraining:
            # When performing pretraining and linear evaluation together we do not
            # want information from linear eval flowing back into pretraining network
            # so we put a stop_gradient.
            supervised_head_outputs = self.supervised_head(
                tf.stop_gradient(supervised_head_inputs), training)

            return obj, back, projection_head_outputs, supervised_head_outputs

        else:
            return obj, back, projection_head_outputs, None

        return obj, back, feature_map_upsample
        # return feature_map_upsample
