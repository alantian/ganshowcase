#!/usr/bin/env python3
'''
Some reference:

 - GP in keras:
   https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
 - GANs in Chainer:
   https://github.com/pfnet-research/chainer-gan-lib

'''

from functools import partial
import os
import math

from absl import app
from absl import flags
from absl import logging
import keras
from keras.layers.core import Activation
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers.merge import _Merge
from keras.layers.convolutional import Convolution2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras import backend as K
import numpy as np
from PIL import Image
from tqdm import tqdm

# WARNING: multiple gpu not supported for wgan-gp


def get_dcgan64_generate(input_dim=100, ch=512, weight_scale=0.02):
    model = Sequential()
    print('weight_scale', weight_scale)
    weight_init = keras.initializers.RandomNormal(stddev=weight_scale)
    model.add(Dense(input_dim=input_dim, units=4 * 4 * ch, kernel_initializer=weight_init))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Reshape((ch, 4, 4)))
    model.add(
        Conv2DTranspose(
            ch // 2,
            kernel_size=4,
            strides=2,
            padding='same',
            data_format='channels_first',
            kernel_initializer=weight_init))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(
        Conv2DTranspose(
            ch // 4,
            kernel_size=4,
            strides=2,
            padding='same',
            data_format='channels_first',
            kernel_initializer=weight_init))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(
        Conv2DTranspose(
            ch // 8,
            kernel_size=4,
            strides=2,
            padding='same',
            data_format='channels_first',
            kernel_initializer=weight_init))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(
        Conv2DTranspose(
            3, kernel_size=4, strides=2, padding='same', data_format='channels_first', kernel_initializer=weight_init))
    model.add(Activation('tanh'))
    return model


def get_dcgan64_discriminator(ch=512, weight_scale=0.02):
    model = Sequential()
    print('weight_scale', weight_scale)
    weight_init = keras.initializers.RandomNormal(stddev=weight_scale)
    model.add(
        Conv2D(
            ch // 8,
            kernel_size=4,
            strides=2,
            padding='same',
            input_shape=(3, 64, 64),
            data_format='channels_first',
            kernel_initializer=weight_init))
    model.add(LeakyReLU(alpha=0.2))
    model.add(
        Conv2D(
            ch // 4,
            kernel_size=4,
            strides=2,
            padding='same',
            data_format='channels_first',
            kernel_initializer=weight_init))
    model.add(BatchNormalization(scale=False, axis=1))
    model.add(LeakyReLU(alpha=0.2))
    model.add(
        Conv2D(
            ch // 4,
            kernel_size=3,
            strides=1,
            padding='same',
            data_format='channels_first',
            kernel_initializer=weight_init))
    model.add(BatchNormalization(scale=False, axis=1))
    model.add(LeakyReLU(alpha=0.2))
    model.add(
        Conv2D(
            ch // 2,
            kernel_size=4,
            strides=2,
            padding='same',
            data_format='channels_first',
            kernel_initializer=weight_init))
    model.add(BatchNormalization(scale=False, axis=1))
    model.add(LeakyReLU(alpha=0.2))
    model.add(
        Conv2D(
            ch // 2,
            kernel_size=3,
            strides=1,
            padding='same',
            data_format='channels_first',
            kernel_initializer=weight_init))
    model.add(BatchNormalization(scale=False, axis=1))
    model.add(LeakyReLU(alpha=0.2))
    model.add(
        Conv2D(
            ch // 1,
            kernel_size=4,
            strides=2,
            padding='same',
            data_format='channels_first',
            kernel_initializer=weight_init))
    model.add(BatchNormalization(scale=False, axis=1))
    model.add(LeakyReLU(alpha=0.2))
    model.add(
        Conv2D(
            ch // 1,
            kernel_size=3,
            strides=1,
            padding='same',
            data_format='channels_first',
            kernel_initializer=weight_init))
    model.add(BatchNormalization(scale=False, axis=1))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((ch * 4 * 4, )))
    model.add(Dense(input_dim=4 * 4 * ch, units=1, kernel_initializer=weight_init))

    return model


def dcgan_loss_real(y_true, y_pred):
    return K.mean(K.softplus(-y_pred))


def dcgan_loss_fake(y_true, y_pred):
    return K.mean(K.softplus(y_pred))


def loss_l2(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))


def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.
    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the loss function
    that penalizes the network if the gradient norm moves away from 1. However, it is impossible to evaluate
    this function at all points in the input space. The compromise used in the paper is to choose random points
    on the lines between real and generated samples, and check the gradients at these points. Note that it is the
    gradient w.r.t. the input averaged samples, not the weights of the discriminator, that we're penalizing!
    In order to evaluate the gradients, we must first run samples through the generator and evaluate the loss.
    Then we get the gradients of the discriminator w.r.t. the input averaged samples.
    The l2 norm and penalty can then be calculated for this gradient.
    Note that this loss function requires the original averaged samples as input, but Keras only supports passing
    y_true and y_pred to loss functions. To get around this, we make a partial() of the function with the
    averaged_samples argument, and use that for model training."""
    # first get the gradients:
    #   assuming: - that y_pred has dimensions (batch_size, 1)
    #             - averaged_samples has dimensions (batch_size, nbr_features)
    # gradients afterwards has dimension (batch_size, nbr_features), basically
    # a list of nbr_features-dimensional gradient vectors
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1. - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)


def tile_images(image_stack):
    """Given a stacked tensor of images, reshapes them into a horizontal tiling for display."""
    assert len(image_stack.shape) == 4
    image_list = [image_stack[i, :, :, :] for i in range(image_stack.shape[0])]
    tiled_images = np.concatenate(image_list, axis=1)
    return tiled_images


def generate_images(generator_model, output_dir, iter, latent_dim, nb_row=5, nb_col=5):
    """Feeds random seeds into the generator and tiles and saves the output to a PNG file."""
    test_image_stack = generator_model.predict(np.random.randn(nb_row * nb_col, latent_dim))
    test_image_stack = (test_image_stack * 127.5) + 127.5
    test_image_stack = np.round(test_image_stack).astype(np.uint8)

    arr = test_image_stack
    _, C, H, W = arr.shape
    arr = np.reshape(arr, (nb_row, nb_col, C, H, W))  # rc * C * H * W -> r * c * C * H * W
    arr = arr.transpose(0, 3, 1, 4, 2)
    arr = np.reshape(arr, (nb_row * H, nb_col * W, C))

    tiled_output = Image.fromarray(arr, mode='RGB')
    outfile = os.path.join(output_dir, 'iter_%08d.png' % iter)
    tiled_output.save(outfile)


def dcgan_sanity_check():
    generate = get_dcgan64_generate()
    discriminator = get_dcgan64_discriminator()

    generate.compile(loss='mean_squared_error', optimizer='sgd')
    discriminator.compile(loss='mean_squared_error', optimizer='sgd')
    batch_size = 7
    hidden_dim = 100
    z = np.random.randn(batch_size, hidden_dim)
    x = generate.predict(z)
    y = discriminator.predict(x)

    print(z.shape)
    print(x.shape)
    print(y.shape)


FLAGS = flags.FLAGS

flags.DEFINE_string('npz_path', '/scratch/celeba/celeba_64.npz', 'Path to images in npz file.')
flags.DEFINE_string('out', '/scratch/model/dcgan-celeba_64', '')
flags.DEFINE_integer('size', 64, 'Size for images')
flags.DEFINE_integer('batch_size', 64, 'Batch size for training')
flags.DEFINE_integer('nb_epoch', 100, 'Number of epoches')
flags.DEFINE_integer('latent_dim', 128, 'Dimension for latent vector.')
flags.DEFINE_integer('sample_every_iter', 10000, '')
flags.DEFINE_float('weight_scale', 0.02, '')
flags.DEFINE_string('gptype', 'DRAGAN', 'Gradient penalty type. Can be WGAN-GP or DRAGAN')


class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this outputs a random point on the line
    between each pair of input points.
    Inheriting from _Merge is a little messy but it was the quickest solution I could think of.
    Improvements appreciated."""

    def _merge_function(self, inputs):
        if FLAGS.gptype == 'WGAN-GP':
            # WGAN-GP start
            batch_size = K.shape(inputs[0])[0]
            weights = K.random_uniform((batch_size, 1, 1, 1))
            output = (weights * inputs[0]) + ((1 - weights) * inputs[1])
        elif FLAGS.gptype == 'DRAGAN':
            # DRAGAN start
            std = K.std(inputs[0], axis=0, keepdims=True)
            weights = K.random_uniform(K.shape(inputs[0]))
            output = inputs[0] + 0.5 * weights * std
            # DRAGAN end
        else:
            raise ValueError('Unknown gptype %s' % FLAGS.gptype)
        return output


def main(argv):
    del argv  # Unused.

    X_train = np.load(FLAGS.npz_path)['size_%d' % FLAGS.size]
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    logging.info('Data loaded shape: %s', X_train.shape)
    '''
    x = X_train[0]
    x = x.transpose(1, 2, 0)
    x = x * 127.5 + 127.5
    x = np.round(x).astype(np.uint8)

    img = Image.fromarray(x, mode='RGB')
    img.save('x.png')
    return
    '''

    GRADIENT_PENALTY_WEIGHT = 1.0
    BATCH_SIZE = FLAGS.batch_size
    NB_EPOCH = FLAGS.nb_epoch
    OUT_DIR = FLAGS.out
    SAMPLE_EVERY_ITER = FLAGS.sample_every_iter
    LATENT_DIM = FLAGS.latent_dim

    os.system('mkdir -p %s' % OUT_DIR)

    generator = get_dcgan64_generate(LATENT_DIM, weight_scale=FLAGS.weight_scale)
    discriminator = get_dcgan64_discriminator(weight_scale=FLAGS.weight_scale)

    # The generator_model is used when we want to train the generator layers.
    # As such, we ensure that the discriminator layers are not trainable.
    # Note that once we compile this model, updating .trainable will have no effect within it. As such, it
    # won't cause problems if we later set discriminator.trainable = True for the discriminator_model, as long
    # as we compile the generator_model first.
    for layer in discriminator.layers:
        layer.trainable = False
    discriminator.trainable = False
    generate_input = Input(shape=(LATENT_DIM, ))
    generate_layers = generator(generate_input)
    discriminator_layers_for_generate = discriminator(generate_layers)
    generator_model = Model(inputs=[generate_input], outputs=[discriminator_layers_for_generate])
    # We use the Adam paramaters from Gulrajani et al.
    generator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.999), loss=dcgan_loss_real, metrics=[])

    # Now that the generator_model is compiled, we can make the discriminator layers trainable.
    for layer in discriminator.layers:
        layer.trainable = True
    for layer in generator.layers:
        layer.trainable = False
    discriminator.trainable = True
    generator.trainable = False

    # The discriminator_model is more complex. It takes both real image samples and random noise seeds as input.
    # The noise seed is run through the generator model to get generated images. Both real and generated images
    # are then run through the discriminator. Although we could concatenate the real and generated images into a
    # single tensor, we don't (see model compilation for why).
    real_samples = Input(shape=X_train.shape[1:])
    generator_input_for_discriminator = Input(shape=(LATENT_DIM, ))
    generated_samples_for_discriminator = generator(generator_input_for_discriminator)
    discriminator_output_from_generator = discriminator(generated_samples_for_discriminator)
    discriminator_output_from_real_samples = discriminator(real_samples)

    # We also need to generate weighted-averages of real and generated samples, to use for the gradient norm penalty.
    averaged_samples = RandomWeightedAverage()([real_samples, generated_samples_for_discriminator])
    # We then run these samples through the discriminator as well. Note that we never really use the discriminator
    # output for these samples - we're only running them to get the gradient norm for the gradient penalty loss.
    averaged_samples_out = discriminator(averaged_samples)

    # The gradient penalty loss function requires the input averaged samples to get gradients. However,
    # Keras loss functions can only have two arguments, y_true and y_pred. We get around this by making a partial()
    # of the function with the averaged samples here.
    partial_gp_loss = partial(
        gradient_penalty_loss, averaged_samples=averaged_samples, gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
    partial_gp_loss.__name__ = 'gradient_penalty'  # Functions need names or Keras will throw an error

    # Keras requires that inputs and outputs have the same number of samples. This is why we didn't concatenate the
    # real samples and generated samples before passing them to the discriminator: If we had, it would create an
    # output with 2 * BATCH_SIZE samples, while the output of the "averaged" samples for gradient penalty
    # would have only BATCH_SIZE samples.

    # If we don't concatenate the real and generated samples, however, we get three outputs: One of the generated
    # samples, one of the real samples, and one of the averaged samples, all of size BATCH_SIZE. This works neatly!
    discriminator_model = Model(
        inputs=[real_samples, generator_input_for_discriminator],
        outputs=[discriminator_output_from_real_samples, discriminator_output_from_generator, averaged_samples_out])
    # We use the Adam paramaters from Gulrajani et al. We use the Wasserstein loss for both the real and generated
    # samples, and the gradient penalty loss for the averaged samples.
    discriminator_model.compile(
        optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.999),
        loss=[dcgan_loss_real, dcgan_loss_fake, partial_gp_loss],
        metrics=[])

    # We make three label vectors for training. positive_y is the label vector for real samples, with value 1.
    # negative_y is the label vector for generated samples, with value -1. The dummy_y vector is passed to the
    # gradient_penalty loss function and is not used.
    # *** actually, we don't use them ***
    positive_y = np.ones((BATCH_SIZE, 1), dtype=np.float32)
    negative_y = -positive_y
    dummy_y = np.zeros((BATCH_SIZE, 1), dtype=np.float32)

    # for sample
    last_sample_iter, iter = None, 0
    sample_dir = os.path.join(OUT_DIR, 'sample')
    os.system('mkdir -p %s' % sample_dir)

    # for saving model
    save_model_dir = os.path.join(OUT_DIR, 'model')
    os.system('mkdir -p %s' % save_model_dir)

    # real training
    for epoch in range(NB_EPOCH):
        np.random.shuffle(X_train)
        print("Epoch: ", epoch)
        print("Number of batches: ", int(X_train.shape[0] // BATCH_SIZE))
        discriminator_loss = []
        generator_loss = []
        pbar = tqdm(range(int(X_train.shape[0] // BATCH_SIZE)), unit=' batch')

        for i in pbar:
            if last_sample_iter is None or iter // SAMPLE_EVERY_ITER != last_sample_iter // SAMPLE_EVERY_ITER:
                generate_images(generator, sample_dir, iter // SAMPLE_EVERY_ITER * SAMPLE_EVERY_ITER, LATENT_DIM)
                last_sample_iter = iter

                #x = generator.predict(np.random.randn(1, LATENT_DIM))
                #assert False
                # return  # exit after first sampling...

            image_batch = X_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            iter += len(image_batch)

            noise = np.random.randn(BATCH_SIZE, LATENT_DIM).astype(np.float32)
            discriminator_loss.append(
                discriminator_model.train_on_batch([image_batch, noise], [positive_y, negative_y, dummy_y]))

            generator_loss.append(generator_model.train_on_batch(np.random.rand(BATCH_SIZE, LATENT_DIM), positive_y))

            pbar.set_description('gen loss %s dis loss %s' %
                                 (('%.6f' % generator_loss[-1]),
                                  ('[%s]' % (' '.join(['%.6f' % _ for _ in discriminator_loss[-1]]), ))))

        # Still needs some code to display losses from the generator and discriminator, progress bars, etc.

        print('generator_loss', generator_loss[-1])
        print('discriminator_loss', discriminator_loss[-1])
        generator.save(os.path.join(save_model_dir, 'generator_epoch_%d.h5' % epoch))
        discriminator.save(os.path.join(save_model_dir, 'discriminator_epoch_%d.h5' % epoch))


import pdb, traceback, sys, code  # noqa

if __name__ == '__main__':
    try:
        app.run(main)
    except Exception:  # noqa
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
