import tensorflow as tf
import numpy as np
from functools import partial

conv2d = tf.layers.conv2d
kinit = tf.variance_scaling_initializer(
	mode='fan_out', scale=2., distribution='untruncated_normal')
dense = tf.layers.dense
dinit = tf.random_normal_initializer(stddev=0.01)


_BATCH_NORM_DECAY = 0.99
_BATCH_NORM_EPSILON = 1e-4


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
batchnorm = tf.layers.batch_normalization
bn = partial(
	batchnorm, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON)


def bnrelu(x, *, is_training, gamma_zero=False, relu=True):
  gamma_init = tf.zeros_initializer() if gamma_zero else tf.ones_initializer()
  x = bn(inputs=x, gamma_initializer=gamma_init, training=is_training)
  if relu:
  	x = tf.nn.relu(x)
  return x


def residual_block(x, *, is_training, filters, factor, strides, projection=False):
  shortcut = x
  if projection:
    shortcut = conv_1x1(
        inputs=shortcut, strides=strides, filters=filters * factor)
    shortcut = bnrelu(shortcut, is_training=is_training, relu=False)
  x = conv_1x1(inputs=x, filters=filters, strides=1)
  x = bnrelu(x, is_training=is_training)
  if strides != 1:
    x = pad(x, 1)
    x = conv_3x3_stride2(inputs=x, filters=filters)
  else:
    x = conv_3x3(inputs=x, filters=filters)
  x = bnrelu(x, is_training=is_training)
  x = conv_1x1(inputs=x, filters=filters * factor, strides=1)
  x = bnrelu(x, is_training=is_training, relu=False, gamma_zero=True)
  assert shortcut.shape[1:] == x.shape[1:]
  return tf.nn.relu(x + shortcut)


def residual_stack(x, *, n_blocks, filters, factor, downsample=True):
  for i_block in range(n_blocks):
    x = residual_block(
        x,
        is_training=is_training,
        filters=filters,
        factor=factor,
        strides=2 if ((i_block == 0) and downsample) else 1,
        projection=(i_block == 0))
  return x


def resnet_v1(x, *, , num_classes, resnet_type=None,
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
    x = tf.layers.max_pooling2d(inputs=x, pool_size=3, strides=2,
                                padding='valid')
    endpoints['c1'] = x
    for i_stack, n_blocks, n_filters, factor in zip(
        range(len(blocks)), blocks, filters, factors):
      x = residual_stack(
          x,
          n_blocks=n_blocks,
          filters=n_filters,
          factor=factor,
          downsample=False if i_stack == 0 else True)
      endpoints['c' + str(i_stack + 2)] = x
    x = tf.reduce_mean(x, [1, 2], keepdims=False)
    logits = dense(inputs=x, units=num_classes, kernel_initializer=dinit)
    return endpoints, logits


def group_pointwise(featuremap, *, proj_channels=None, name='grouppoint', heads=8):
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    in_channels = featuremap.shape[-1]
    proj_channels = proj_channels or in_channels // 4
    fan_in = np.prod(featuremap.shape[1:].as_list())
    w = tf.get_variable(
        'w',
        [in_channels, heads, proj_channels // heads],
        dtype=featuremap.dtype,
        initializer=tf.random_normal_initializer(stddev=2./fan_in))
    out = tf.einsum('bHWD,Dhd->bhHWd', featuremap, w) # put heads first.
    return out


def output_proj(featuremap, out_channels=None, name='outputproj'):
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    in_heads, in_channels = featuremap.shape[-2:]
    out_channels = out_channels or in_heads * in_channels
    fan_in = np.prod(featuremap.shape[1:].as_list())
    w = tf.get_variable(
        'w',
        [in_heads, in_channels, out_channels],
        dtype=featuremap.dtype,
        initializer=tf.random_normal_initializer(stddev=2./fan_in))
    out = tf.einsum('bHWhd,hdD->bHWD', featuremap, w)
    return out


def self_attention(q, k, v):
	# bhHWd,bhPQd->bhHWPQ; bhHWPQ,bhPQd->bHWhd
  head_dim = q.shape[-1]
  scale_factor = 1. / math.sqrt(head_dim.value)
  logits = tf.einsum('bhHWd,bhPQd->bhHWPQ', q, k)
  logits = tf.nn.softmax(logits * scale_factor)
  out = tf.einsum('bhHWPQ,bhPQd->bHWhd', logits, v)
  return out


def relpos_self_attention(q, k, v):
  # does not do factorized pos_embs for x, y.
  heads, height, width, dim_head = q.shape.as_list()[1:]
  # q, k, v: [b, h, H, W, d], [b, h, H, W, d], [b, h, H, W, d], r: [h, H, W, d]
  head_dim = q.shape[-1]
  scale_factor = 1. / math.sqrt(head_dim.value)
  r = tf.get_variable(
      'r',
      [height, width, dim_head],
      dtype=q.dtype,
      initializer=tf.random_normal_initializer(stddev=2./dim_head))
  logits = tf.einsum('bhHWd,bhPQd->bhHWPQ', q, k) + tf.einsum('bhHWd,PQd->bhHWPQ', q, r)
  logits = tf.nn.softmax(logits * scale_factor)
  out = tf.einsum('bhHWPQ,bhPQd->bHWhd', logits, v)
  return out


def all2all(featuremap, is_training_bn, heads=8, num_batch_norm_group=None, proj_channels=None, out_channels=None, name='all2all'):
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    in_channels = featuremap.shape[-1]
    q = group_pointwise(featuremap, proj_channels=proj_channels, name='q_proj', heads=heads)
    k = group_pointwise(featuremap, proj_channels=proj_channels, name='k_proj', heads=heads)
    v = group_pointwise(featuremap, proj_channels=proj_channels, name='v_proj', heads=heads)
    o = relpos_self_attention(q, k, v)
    o = output_proj(o, out_channels=out_channels, name='o_proj')
    o = nn_ops.batch_norm_relu(o, is_training_bn, relu=False, init_zero=True, num_batch_norm_group=num_batch_norm_group)
    return featuremap + o

