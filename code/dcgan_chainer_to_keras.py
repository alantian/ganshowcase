#!/usr/bin/env python3

from functools import partial
import os
import sys

from absl import app
from absl import flags
from absl import logging

import keras
from keras import backend as K
from keras.layers.core import Activation
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Reshape, UpSampling2D, Add
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
import numpy as np
from PIL import Image
import tensorflowjs as tfjs


def _make_dense(input_dim, units, weight=None, kernel_arr_name=None, bias_arr_name=None):
    return Dense(
        input_dim=input_dim,
        units=units,
        kernel_initializer=(lambda x: np.transpose(weight[kernel_arr_name], (1, 0))),
        bias_initializer=(lambda x: weight[bias_arr_name]),
    ) if weight else Dense(
        input_dim=input_dim,
        units=units,
    )


def _make_batch_normalizzation(axis,
                               weight=None,
                               beta_arr_name=None,
                               gamma_arr_name=None,
                               moving_mean_arr_name=None,
                               moving_variance_arr_name=None):
    return BatchNormalization(
        axis=1,
        beta_initializer=(lambda x: weight[beta_arr_name]),
        gamma_initializer=(lambda x: weight[gamma_arr_name]),
        moving_mean_initializer=(lambda x: weight[moving_mean_arr_name]),
        moving_variance_initializer=(lambda x: weight[moving_variance_arr_name]),
    ) if weight else BatchNormalization(axis=axis)


def _make_conv_2d_transpose(filters, kernel_size, strides, weight=None, kernel_arr_name=None, bias_arr_name=None):

    return Conv2DTranspose(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        data_format='channels_first',
        kernel_initializer=(lambda x: np.transpose(weight[kernel_arr_name], (2, 3, 1, 0))),
        bias_initializer=(lambda x: weight[bias_arr_name]),
    ) if weight else Conv2DTranspose(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        data_format='channels_first',
    )


def _make_conv_2d(filters, kernel_size, strides, weight=None, kernel_arr_name=None, bias_arr_name=None):

    return Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        data_format='channels_first',
        kernel_initializer=(lambda x: np.transpose(weight[kernel_arr_name], (2, 3, 1, 0))),
        bias_initializer=(lambda x: weight[bias_arr_name]),
    ) if weight else Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        data_format='channels_first',
    )


def get_dcgan64_keras_generator(input_dim, ch, weight=None):
    if weight:
        print('=' * 80)
        print('weight')
        print('-' * 80)
        keys = list(sorted(weight.keys()))

        for key in keys:
            print(key, weight[key].shape)
        print('=' * 80)

    model = Sequential()
    model.add(_make_dense(input_dim, 4 * 4 * ch, weight, 'l0/W', 'l0/b'))
    model.add(_make_batch_normalizzation(1, weight, 'bn0/beta', 'bn0/gamma', 'bn0/avg_mean', 'bn0/avg_var'))
    model.add(Activation('relu'))
    model.add(Reshape((ch, 4, 4)))
    model.add(_make_conv_2d_transpose(ch // 2, 4, 2, weight, 'dc1/W', 'dc1/b'))
    model.add(_make_batch_normalizzation(1, weight, 'bn1/beta', 'bn1/gamma', 'bn1/avg_mean', 'bn1/avg_var'))
    model.add(Activation('relu'))
    model.add(_make_conv_2d_transpose(ch // 4, 4, 2, weight, 'dc2/W', 'dc2/b'))
    model.add(_make_batch_normalizzation(1, weight, 'bn2/beta', 'bn2/gamma', 'bn2/avg_mean', 'bn2/avg_var'))
    model.add(Activation('relu'))
    model.add(_make_conv_2d_transpose(ch // 8, 4, 2, weight, 'dc3/W', 'dc3/b'))
    model.add(_make_batch_normalizzation(1, weight, 'bn3/beta', 'bn3/gamma', 'bn3/avg_mean', 'bn3/avg_var'))
    model.add(Activation('relu'))
    model.add(_make_conv_2d_transpose(3, 4, 2, weight, 'dc4/W', 'dc4/b'))
    model.add(Activation('tanh'))
    return model


def _make_upsampling_2d(ch, weight=None):
    '''
    tensorflow.js and Keras doesn't support Unpooling,
    use Conv2DTranspose with hanf-crafted weight as a replacement.
    '''

    kernel_matrix = np.zeros((2, 2, ch, ch), dtype='f')
    for i in range(ch):
        kernel_matrix[:, :, i, i] = 1.

    return Conv2DTranspose(
        filters=ch,
        kernel_size=2,
        strides=2,
        padding='same',
        data_format='channels_first',
        # This is a trick.
        # When weight is None, we can leave any standard initilizer
        # since weight would be later loaded, but we cannot leave any lambda
        # that tfjs would complain when it loads its weight.
        kernel_initializer=(lambda x: kernel_matrix) if weight else keras.initializers.Zeros(),
        bias_initializer=keras.initializers.Zeros())


def _make_res_net_res_block_up(in_ch, out_ch, weight=None, prefix=''):
    def f(x):
        p = prefix
        bn0 = _make_batch_normalizzation(1, weight, p + 'bn0/beta', p + 'bn0/gamma', p + 'bn0/avg_mean',
                                         p + 'bn0/avg_var')
        bn1 = _make_batch_normalizzation(1, weight, p + 'bn1/beta', p + 'bn1/gamma', p + 'bn1/avg_mean',
                                         p + 'bn1/avg_var')
        c0 = _make_conv_2d(out_ch, 3, 1, weight, p + 'c0/W', p + 'c0/b')
        c1 = _make_conv_2d(out_ch, 3, 1, weight, p + 'c1/W', p + 'c1/b')
        cs = _make_conv_2d(out_ch, 3, 1, weight, p + 'cs/W', p + 'cs/b')

        u0 = _make_upsampling_2d(in_ch, weight)
        u1 = _make_upsampling_2d(in_ch, weight)

        h = c0(u0(Activation('relu')(bn0(x))))
        h = c1(Activation('relu')(bn1(h)))
        hs = cs(u1(x))
        return Add()([h, hs])

    return f


def _make_rese_net_finals(ch, weight=None, prefix=''):
    def f(x):
        p = prefix

        bn = _make_batch_normalizzation(1, weight, p + '0/beta', p + '0/gamma', p + '0/avg_mean', p + '0/avg_var')
        c = _make_conv_2d(ch, 3, 1, weight, p + '2/W', p + '2/b')

        h = Activation('relu')(bn(x))
        h = Activation('tanh')(c(h))

        return h

    return f


def get_resnet128_keras_generator(input_dim, ch, weight=None):
    if weight:
        print('=' * 80)
        print('weight')
        print('-' * 80)
        keys = list(sorted(weight.keys()))

        for key in keys:
            print(key, weight[key].shape)
        print('=' * 80)

    input = Input(shape=(input_dim, ))
    x = input
    x = _make_dense(input_dim, 4 * 4 * ch, weight, 'dense/l/W', 'dense/l/b')(x)
    x = Reshape((ch, 4, 4))(x)
    x = _make_res_net_res_block_up(ch, ch, weight, 'resblockups/0/')(x)
    x = _make_res_net_res_block_up(ch, ch // 2, weight, 'resblockups/1/')(x)
    x = _make_res_net_res_block_up(ch // 2, ch // 4, weight, 'resblockups/2/')(x)
    x = _make_res_net_res_block_up(ch // 4, ch // 8, weight, 'resblockups/3/')(x)
    x = _make_res_net_res_block_up(ch // 8, ch // 16, weight, 'resblockups/4/')(x)
    x = _make_rese_net_finals(3, weight, 'finals/')(x)

    model = Model(inputs=input, outputs=x)
    return model


def get_resnet256_keras_generator(input_dim, ch, weight=None):
    if weight:
        print('=' * 80)
        print('weight')
        print('-' * 80)
        keys = list(sorted(weight.keys()))

        for key in keys:
            print(key, weight[key].shape)
        print('=' * 80)

    input = Input(shape=(input_dim, ))
    x = input
    x = _make_dense(input_dim, 4 * 4 * ch, weight, 'dense/l/W', 'dense/l/b')(x)
    x = Reshape((ch, 4, 4))(x)
    x = _make_res_net_res_block_up(ch, ch, weight, 'resblockups/0/')(x)
    x = _make_res_net_res_block_up(ch, ch // 2, weight, 'resblockups/1/')(x)
    x = _make_res_net_res_block_up(ch // 2, ch // 4, weight, 'resblockups/2/')(x)
    x = _make_res_net_res_block_up(ch // 4, ch // 8, weight, 'resblockups/3/')(x)
    x = _make_res_net_res_block_up(ch // 8, ch // 16, weight, 'resblockups/4/')(x)
    x = _make_res_net_res_block_up(ch // 16, ch // 32, weight, 'resblockups/5/')(x)
    x = _make_rese_net_finals(3, weight, 'finals/')(x)

    model = Model(inputs=input, outputs=x)
    return model


def generate_images(generator, output_dir, index, latent_dim=128, nb_row=5, nb_col=5):
    """Feeds random seeds into the generator and tiles and saves the output to a PNG file."""
    test_image_stack = generator.predict(np.random.randn(nb_row * nb_col, latent_dim))
    test_image_stack = (test_image_stack * 127.5) + 127.5
    test_image_stack = np.round(test_image_stack).astype(np.uint8)

    arr = test_image_stack
    _, C, H, W = arr.shape
    arr = np.reshape(arr, (nb_row, nb_col, C, H, W))  # rc * C * H * W -> r * c * C * H * W
    arr = arr.transpose(0, 3, 1, 4, 2)
    arr = np.reshape(arr, (nb_row * H, nb_col * W, C))

    tiled_output = Image.fromarray(arr, mode='RGB')
    outfile = os.path.join(output_dir, '%08d.png' % index)
    tiled_output.save(outfile)


FLAGS = flags.FLAGS

flags.DEFINE_string('arch', '', 'Architecture of netowrk. can be `dcgan64` or `resnet128`.')
flags.DEFINE_string('chainer_model_path', '', '')
flags.DEFINE_string('keras_model_path', '', '')
flags.DEFINE_string('tfjs_model_path', '', '')


def main(argv):
    del argv  # Unused.

    weight = np.load(FLAGS.chainer_model_path)

    if FLAGS.arch == 'resnet128':
        get_generator = partial(get_resnet128_keras_generator, input_dim=128, ch=1024)
    elif FLAGS.arch == 'resnet256':
        get_generator = partial(get_resnet256_keras_generator, input_dim=128, ch=1024)
    elif FLAGS.arch == 'dcgan64':
        get_generator = partial(get_dcgan64_keras_generator, input_dim=128, ch=512)
    else:
        raise ValueError('Unknow --arch %s' % FLAGS.arch)

    generator = get_generator(weight=weight)
    print('Keras summary')
    generator.summary()
    logging.info('Saving keras model (weights) to %s', FLAGS.keras_model_path)
    generator.save_weights(FLAGS.keras_model_path)
    del generator
    # this avoids lambda initilizers in generator, which whould cause error in tfjs.
    generator = get_generator()
    generator.load_weights(FLAGS.keras_model_path)
    generator.save_weights(FLAGS.keras_model_path)

    logging.info('Saving tensorflow.js model to %s', FLAGS.tfjs_model_path)
    os.system('mkdir -p "%s"' % FLAGS.tfjs_model_path)
    tfjs.converters.save_keras_model(generator, FLAGS.tfjs_model_path)

    sample_output_dir = FLAGS.keras_model_path + '.sample'
    logging.info('Sampling images, saving to %s', sample_output_dir)
    os.system('mkdir -p "%s"' % sample_output_dir)
    for index in range(10):
        generate_images(generator, sample_output_dir, index)


import pdb, traceback, sys, code  # noqa

if __name__ == '__main__':
    try:
        app.run(main)
    except Exception:  # noqa
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
