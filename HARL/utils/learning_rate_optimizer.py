'''
Strategy Training Need to Implementation Effective
# In Contrastive SSL framework
************************************************************************************
Training Configure
************************************************************************************
1. Learning Rate
    + particular implementation : Scale Learning Rate Linearly with Batch_SIZE (With Cosine Learning Rate)
    + Warmup: Learning Implementation
    + Schedule Learning with Constrain-Update during training

2. Optimizer -- With & Without Gradient Centralize
    1.LARS_optimizer for Contrastive + Large batch_size
    2. SGD - RMSProp - Adam (Gradient Centralize)
    3. SGD -- RMSProp -- Adam (Weight Decay) (TFA)

3. Regularization Weight Decay
    weight decay: Start with 1e6

************************************************************************************
FineTuning Configure
************************************************************************************
1. Learning Rate

2. Optimizer (Regularization weight Decay)

'''

# 1 implementation Cosine Decay Learning rate schedule (Warmup Period)-- Implement
'''
[1] SimCLR [2] BYOL [Consisten Distillation Training]
in SimCLR & BYOL with warmup steps =10
'''

# Reference https://github.com/google-research/simclr/blob/dec99a81a4ceccb0a5a893afecbc2ee18f1d76c3/tf2/model.py


#import tensorflow.compat.v2 as tf
#import tensorflow_addons as tfa
#import tensorflow as tf


# this helper function determine steps per epoch




from  HARL.utils.lars_optimizer import LARSOptimizer as LARS_optimzer
import tensorflow.keras.backend as K
import numpy as np
import math
import tensorflow as tf
import tensorflow_addons as tfa
from math import floor, cos, pi
def get_centralized_gradients(optimizer, loss, params):
    """Compute the centralized gradients.

    This function is ideally not meant to be used directly unless you are building a custom optimizer, in which case you
    could point `get_gradients` to this function. This is a modified version of
    `tf.keras.optimizers.Optimizer.get_gradients`.

    # Arguments:
        optimizer: a `tf.keras.optimizers.Optimizer object`. The optimizer you are using.
        loss: Scalar tensor to minimize.
        params: List of variables.

    # Returns:
      A gradients tensor.

    # Reference:
        [Yong et al., 2020](https://arxiv.org/abs/2004.01461)
    """

    # We here just provide a modified get_gradients() function since we are trying to just compute the centralized
    # gradients at this stage which can be used in other optimizers.
    grads = []
    for grad in K.gradients(loss, params):

        grad_len = len(grad.shape)
        if grad_len > 1:
            axis = list(range(grad_len - 1))
            grad -= tf.reduce_mean(grad,
                                   axis=axis,
                                   keep_dims=True)
        grads.append(grad)

    if None in grads:
        raise ValueError('An operation has `None` for gradient. '
                         'Please make sure that all of your ops have a '
                         'gradient defined (i.e. are differentiable). '
                         'Common ops without gradient: '
                         'K.argmax, K.round, K.eval.')
    if hasattr(optimizer, 'clipnorm') and optimizer.clipnorm > 0:
        norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
        grads = [
            tf.keras.optimizers.clip_norm(
                g,
                optimizer.clipnorm,
                norm) for g in grads]
    if hasattr(optimizer, 'clipvalue') and optimizer.clipvalue > 0:
        grads = [K.clip(g, -optimizer.clipvalue, optimizer.clipvalue)
                 for g in grads]
    return grads


def centralized_gradients_for_optimizer(optimizer):
    """Create a centralized gradients functions for a specified optimizer.

    # Arguments:
        optimizer: a `tf.keras.optimizers.Optimizer object`. The optimizer you are using.

    # Usage:

    ```py
    >>> opt = tf.keras.optimizers.Adam(learning_rate=0.1)
    >>> opt.get_gradients = gctf.centralized_gradients_for_optimizer(opt)
    >>> model.compile(optimizer = opt, ...)
    ```
    """

    def get_centralized_gradients_for_optimizer(loss, params):
        return get_centralized_gradients(optimizer, loss, params)

    return get_centralized_gradients_for_optimizer


def get_train_steps(num_examples, train_epochs, gloabl_batch_size, train_steps=None):
    """Determine the number of training steps."""
    if train_steps is None:
        print("Calculate training steps")
        train_steps = (num_examples * train_epochs //
                       gloabl_batch_size + 1)
    else:
        print("You Implement the args training steps")
        train_steps = train_steps

    return train_steps


'''
********************************************
Training Configure
********************************************
1. Warmup Cosine Decay Learning Rate with 1 cycle 
    + particular implementation : Scale Learning Rate Linearly with Batch_SIZE 
    (Warmup: Learning Implementation, and Cosine Anealing + Linear scaling)
   
    # optional not implement yet
    + Schedule Learning with Constrain-Update during training

2. 
'''
# Implementation form SimCLR paper (Linear Scale and Sqrt Scale)
# Debug and Visualization
# Section SimCLR Implementation Learning rate BYOL implementation
# https://colab.research.google.com/drive/1MWgcDAqnB0zZlXz3fHIW0HLKwZOi5UBb?usp=sharing


class WarmUpAndCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Applies a warmup schedule on a given learning rate decay schedule.
    Args:
    Base Learning Rate: is maximum learning Archieve (change with scale applied)
    num_example
    """

    def __init__(self, base_learning_rate, batch_size, num_examples,
                 learning_rate_scale, warmup_epochs, train_epochs, train_steps=None, name=None):
        super(WarmUpAndCosineDecay, self).__init__()
        self.base_learning_rate = base_learning_rate
        self.num_examples = num_examples
        self.Batch_size = batch_size
        self.learning_rate_scale = learning_rate_scale
        self.warmup_epochs = warmup_epochs
        self.train_epochs = train_epochs
        self.train_steps = train_steps
        self._name = name

    def __call__(self, step):
        with tf.name_scope(self._name or 'WarmUpAndCosineDecay'):

            warmup_steps = int(
                round(self.warmup_epochs * self.num_examples //
                      self.Batch_size))
            if self.learning_rate_scale == 'linear':
                scaled_lr = self.base_learning_rate * self.Batch_size / 256.
            elif self.learning_rate_scale == 'sqrt':
                scaled_lr = self.base_learning_rate * \
                    math.sqrt(self.Batch_size)
            elif self.learning_rate_scale == 'no_scale':
                scaled_lr = self.base_learning_rate
            else:
                raise ValueError('Unknown learning rate scaling {}'.format(
                    self.learning_rate_scale))
            learning_rate = (
                step / float(warmup_steps) * scaled_lr if warmup_steps else scaled_lr)

            # Cosine decay learning rate schedule
            total_steps = get_train_steps(
                self.num_examples, self.train_epochs, self.Batch_size, self.train_steps)
            # TODO(srbs): Cache this object.
            cosine_decay = tf.keras.experimental.CosineDecay(
                scaled_lr, total_steps - warmup_steps)
            learning_rate = tf.where(step < warmup_steps, learning_rate,
                                     cosine_decay(step - warmup_steps))

            return learning_rate


class CosineAnnealingDecayRestarts(tf.keras.optimizers.schedules.LearningRateSchedule):
    """A LearningRateSchedule that uses a cosine decay schedule with restarts.
    See [Loshchilov & Hutter, ICLR2016](https://arxiv.org/abs/1608.03983),
    SGDR: Stochastic Gradient Descent with Warm Restarts.
    """

    def __init__(self, initial_learning_rate, first_decay_steps, batch_size, scale_lr,
                 # Control cycle of next step base of Previous step (2 times more steps)
                 t_mul=2.0,
                 # Control ititial Learning Rate Values (Next step equal to previous steps)
                 m_mul=1.0,
                 alpha=0.0,  # Final values of learning rate cycle
                 name=None):
        """Applies cosine decay with restarts to the learning rate.
        Args:
        initial_learning_rate: A scalar `float32` or `float64` Tensor or a Python
            number. The initial learning rate.
        first_decay_steps: A scalar `int32` or `int64` `Tensor` or a Python
            number. Number of steps to decay over.
        t_mul: A scalar `float32` or `float64` `Tensor` or a Python number.
            Used to derive the number of iterations in the i-th period.
        m_mul: A scalar `float32` or `float64` `Tensor` or a Python number.
            Used to derive the initial learning rate of the i-th period.
        alpha: A scalar `float32` or `float64` Tensor or a Python number.
            Minimum learning rate value as a fraction of the initial_learning_rate.
        name: String. Optional name of the operation.  Defaults to 'SGDRDecay'.
        """
        super(CosineAnnealingDecayRestarts, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.first_decay_steps = first_decay_steps
        self._t_mul = t_mul
        self._m_mul = m_mul
        self.alpha = alpha
        self.Batch_size= batch_size
        self.learning_rate_scale = scale_lr
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "SGDRDecay") as name:

            if self.learning_rate_scale == 'linear':
                scaled_lr = self.initial_learning_rate * self.Batch_size / 256.
            elif self.learning_rate_scale == 'sqrt':
                scaled_lr = self.initial_learning_rate * \
                    math.sqrt(self.Batch_size)
            elif self.learning_rate_scale == 'no_scale':
                scaled_lr = self.initial_learning_rate
            else:
                raise ValueError('Unknown learning rate scaling {}'.format(
                    self.learning_rate_scale))

            initial_learning_rate = tf.convert_to_tensor(
                scaled_lr, name="initial_scale_learning_rate")
            dtype = initial_learning_rate.dtype
            first_decay_steps = tf.cast(self.first_decay_steps, dtype)
            alpha = tf.cast(self.alpha, dtype)
            t_mul = tf.cast(self._t_mul, dtype)
            m_mul = tf.cast(self._m_mul, dtype)

            global_step_recomp = tf.cast(step, dtype)
            completed_fraction = global_step_recomp / first_decay_steps

            def compute_step(completed_fraction, geometric=False):
                """Helper for `cond` operation."""
                if geometric:
                    i_restart = tf.floor(
                        tf.math.log(1.0 - completed_fraction * (1.0 - t_mul)) /
                        tf.math.log(t_mul))

                    sum_r = (1.0 - t_mul**i_restart) / (1.0 - t_mul)
                    completed_fraction = (completed_fraction -
                                          sum_r) / t_mul**i_restart

                else:
                    i_restart = tf.floor(completed_fraction)
                    completed_fraction -= i_restart

                return i_restart, completed_fraction

            i_restart, completed_fraction = tf.cond(
                tf.equal(t_mul, 1.0),
                lambda: compute_step(completed_fraction, geometric=False),
                lambda: compute_step(completed_fraction, geometric=True))

            m_fac = m_mul**i_restart
            cosine_decayed = 0.5 * m_fac * (1.0 + tf.cos(
                tf.constant(math.pi, dtype=dtype) * completed_fraction))
            decayed = (1 - alpha) * cosine_decayed + alpha

            return tf.multiply(initial_learning_rate, decayed, name=name)


'''
********************************************
Training Configure
********************************************
2. Optimizer Strategy + Regularization (Weight Decay)
    The optimizer will have three Options
    1. Orginal 
    2. Implmentation with Weight Decay
    3. Implementation with Gradient Centralization
    4  Implementation with Weight Decay and Gradient Centralization 
    ## Optional Consider Clip_Norm strategy
'''


class get_optimizer():
    '''
    The optimizer will have three Options
    1. Orginal 
    2. Implmentation with Weight Decay
    3. Implementation with Gradient Centralization
    4  Implementation with Weight Decay and Gradient Centralization 

    ## Optional Consider Clip_Norm strategy

    '''

    def __init__(self, learning_rate, optimizer_option):
        self.learning_rate = learning_rate
        self.optimizer_ops = optimizer_option

    def original_optimizer(self, args):
        '''Args
          - arsg.optimizer type + Learning rate
          Return Optimizer
        '''
        if self.optimizer_ops == "Adam":
            print("You are implement Adam optimizer")
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate)

        elif self.optimizer_ops == "SGD":
            print("You are implement SGD optimizer")
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=self.learning_rate, momentum=args.momentum,)

        elif self.optimizer_ops == "LARS":
            print("You are implement LARS optimizer")
            optimizer = LARS_optimzer(learning_rate=self.learning_rate,
                                      momentum=args.momentum,
                                      exclude_from_weight_decay=['batch_normalization', 'bias',
                                                                 'head_supervised'])
        return optimizer

    def optimizer_weight_decay(self, args):
        '''Args
          -args.optimizer + args.weight_decay
          Return Optimizer with weight Decay 
        '''
        if self.optimizer_ops == "AdamW":
            print("You are implement Adam Weight decay optimizer")
            optimizer = tfa.optimizers.AdamW(
                weight_decay=args.weight_decay, learning_rate=self.learning_rate)
        if self.optimizer_ops == "SGDW":
            print("You are implement SGD Weight Decay optimizer")
            optimizer = tfa.optimizers.SGDW(
                weight_decay=args.weight_decay, learning_rate=self.learning_rate)

        if self.optimizer_ops == "LARSW":
            print("You are implement LARS weight decay optimizer")
            optimizer = LARS_optimzer(learning_rate=self.learning_rate,
                                      momentum=args.momentum,
                                      weight_decay=args.weight_decay,
                                      exclude_from_weight_decay=['batch_normalization', 'bias',
                                                                 'head_supervised'])
        return optimizer

    def optimizer_gradient_centralization(self, args):
        '''
        Args
        - args.optimizer + Gradient Centralization 
        return Optimizer with Centralization gradient

        '''
        if self.optimizer_ops == 'AdamGC':
            print("You are implement Adam Gradient Centralization optimizer")
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate)
            optimizer.get_gradients = centralized_gradients_for_optimizer(
                optimizer)
        if self.optimizer_ops == "SGDGC":
            print("You are implement SGD Gradient Centralization optimizer")
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=self.learning_rate, momentum=args.momentum)
            optimizer.get_gradients = centralized_gradients_for_optimizer(
                optimizer)
        if self.optimizer_ops == "LARSGC":
            print("You are implement LARS Gradient Centralization optimizer")
            optimizer = LARS_optimzer(learning_rate=self.learning_rate,
                                      momentum=args.momentum,
                                      exclude_from_weight_decay=['batch_normalization', 'bias',
                                                                 'head_supervised'])
            optimizer.get_gradients = centralized_gradients_for_optimizer(
                optimizer)
        return optimizer

    def optimizer_weight_decay_gradient_centralization(self, args):

        if self.optimizer_ops == "AdamW_GC":
            print(
                "You are implement Adam weight decay and Gradient Centralization optimizer")
            optimizer = tfa.optimizers.AdamW(
                weight_decay=args.weight_decay, learning_rate=self.learning_rate)
            optimizer.get_gradients = centralized_gradients_for_optimizer(
                optimizer)

        if self.optimizer_ops == "SGDW_GC":
            print(
                "You are implement SGD weight decay and Gradient Centralization optimizer")
            optimizer = tfa.optimizers.SGDW(
                weight_decay=args.weight_decay, learning_rate=self.learning_rate)
            optimizer.get_gradients = centralized_gradients_for_optimizer(
                optimizer)

        if self.optimizer_ops == "LARSW_GC":
            print(
                "You are implement LARS weight decay and Gradient Centralization optimizer")
            optimizer = LARS_optimzer(learning_rate=self.learning_rate,
                                      momentum=args.momentum,
                                      weight_decay=args.weight_decay,
                                      exclude_from_weight_decay=['batch_normalization', 'bias',
                                                                 'head_supervised'])
            optimizer.get_gradients = centralized_gradients_for_optimizer(
                optimizer)

        return optimizer
