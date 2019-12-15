# Lint as: python3
"""Really concise ResNet."""
import tensorflow as tf
from functools import partial

conv2d = tf.layers.conv2d
dense = tf.layers.dense
kinit = tf.variance_scaling_initializer(scale=2, mode='fan_in',
                                        distribution='untruncated_normal')
dinit = tf.truncated_normal_initializer(stddev=0.01)
conv_1x1 = partial(conv2d, kernel_initializer=kinit, kernel_size=1,
                   use_bias=False, padding='same')
conv_3x3 = partial(conv2d, kernel_initializer=kinit, kernel_size=3, strides=1,
                   use_bias=False, padding='same')
conv_3x3_stride2 = partial(conv2d, kernel_initializer=kinit, kernel_size=3,
                           strides=2, use_bias=False, padding='valid')
conv_7x7_stride2 = partial(conv2d, kernel_initializer=kinit, kernel_size=7,
                           strides=2, use_bias=False, padding='valid')
pad = lambda x, k: tf.pad(x, [[0, 0], [k, k], [k, k], [0, 0]],
                          mode='SYMMETRIC')

groupnorm = tf.contrib.layers.group_norm
gn = partial(groupnorm, groups=32, epsilon=1e-5)


def gnrelu(x, gamma_zero=False):
  gamma_init = tf.zeros_initializer() if gamma_zero else tf.ones_initializer()
  x = gn(inputs=x, param_initializers={'gamma': gamma_init})
  x = tf.nn.relu(x)
  return x


def residual_block(x, *, filters, factor, strides, projection=False):
  shortcut = x
  if projection:
    shortcut = conv_1x1(
        inputs=shortcut, strides=strides, filters=filters * factor)
  x = gnrelu(x)
  x = conv_1x1(inputs=x, filters=filters, strides=1)
  x = gnrelu(x)
  if strides != 1:
    x = pad(x, 1)
    x = conv_3x3_stride2(inputs=x, filters=filters)
  else:
    x = conv_3x3(inputs=x, filters=filters)
  x = gnrelu(x, gamma_zero=True)
  x = conv_1x1(inputs=x, filters=filters * factor, strides=1)
  assert shortcut.shape[1:] == x.shape[1:]
  return x + shortcut


def residual_stack(x, *, n_blocks, filters, factor, downsample=True):
  for i_block in range(n_blocks):
    x = residual_block(
        x,
        filters=filters,
        factor=factor,
        strides=2 if ((i_block == 0) and downsample) else 1,
        projection=(i_block == 0))
  return x


def resnet_v2(x, *, , num_classes, resnet_type=None,
              blocks=None, filters=None, factors=None):
  with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
    type2blocks = {
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
        170: [3, 4, 46, 3],
        200: [3, 24, 36, 3]}
    if resnet_type:
      blocks, filters, factors = type2blocks[resnet_type], [64, 128, 256,
                                                            512], [4, 4, 4, 4]
    else:
      blocks = blocks or [3, 4, 6, 3]
      filters = filters or [64, 128, 256, 512]
      factors = factors or [4, 4, 4, 4]
      # make sure you either use defaults for all or provided args for all
      assert len(blocks) == len(factors) == len(filters)
    endpoints = {}
    x = pad(x, 3)
    x = conv_7x7_stride2(inputs=x, filters=64)
    x = pad(x, 1)
    endpoints['c1'] = x
    x = tf.layers.max_pooling2d(inputs=x, pool_size=3, strides=2,
                                padding='valid')
    for i_stack, n_blocks, n_filters, factor in zip(
        range(len(blocks)), blocks, filters, factors):
      x = residual_stack(
          x,
          norm_fn=norm_fn,
          n_blocks=n_blocks,
          filters=n_filters,
          factor=factor,
          downsample=False if i_stack == 0 else True)
      endpoints['c' + str(i_stack + 2)] = x
    x = tf.reduce_mean(x, [1, 2], keepdims=False)
    logits = dense(inputs=x, units=num_classes, kernel_initializer=dinit)
    return endpoints, logits

