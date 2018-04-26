# GAN Showcase

[Web showcase](https://alantian.net/ganshowcase/) is available.

This is a showcase of a deep GAN (Generative Adversarial Network) that generates (or dreams) images.

Technically, the network architecture is similar to the residual network (ResNet) based generator
([Gulrajani et al.](https://arxiv.org/abs/1704.00028)),
as well as the classical DCGAN generator [Radford et al.](https://arxiv.org/abs/1511.06434)
and the GAN training uses DRAGAN [Kodali et al.](https://arxiv.org/abs/1705.07215)
style gradient penalty for better stability.

Training code is written in[Chainer](https://chainer.org/).
The trained model is then manually converted to a [Keras](https://keras.io/") model,
which in turn is [converted](https://js.tensorflow.org/tutorials/import-keras.html")
to a web-runnable [TensorFlow.js](https://js.tensorflow.org/) model.

The dataset used for training is CelebAHQ, an dataset for [Karras et al.](https://openreview.net/forum?id=Hk99zCeAb&noteId=ryOnMk6rM)
which can be obtained by consulting its GitHub repo (https://github.com/tkarras/progressive_growing_of_gans).


## What is in This Repo

This repo consists of code that

1. Prepares data
2. trains deep GAN.
3. Converts the saved model to a web-runnable one.
4. Presents the deep neural network, running completely in the browser.

## How to use the code

Step 1 through 3 are done offline and are covered by scripts in `./code/` directory and their discussing assumes that you are in `./code/` directory.
Also note several bash variables in `UPPERCASE` should be adjusted
accordingly.

### Step 1 - Prepare data

Dataset is stored as an [npz](https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html) file, and can be converted from either a folder containing images or the CelebAHQ dataset.

For using the folder of images, use

```
DIR_PATH=...
DATA_FILE=...
SIZE=... # can be 64, 128 or 256
./datatool.py --task dir_to_npz \
  --dir_path $DIR_PATH --npz_path $DATA_FILE --size $SIZE
```

For using the CelebAHQ dataset which can be obtained by consulting [its GitHub repo](https://github.com/tkarras/progressive_growing_of_gans):


```
CELEBAHQ_PATH=...  # should be an h5 file
DATA_FILE=...
SIZE=... # can be 64, 128 or 256

../scripts/run_docker.sh \
./datatool.py --task multisize_h5_to_npz \
  --multisize_h5_path $CELEBAHQ_PATH  --npz_path $DATA_FILE --size $SIZE
```

### Step 2 - Training the model

```

# Training DCGAN64 model
DATA_FILE_SIZE_64=...  # Data file of size 64
DCGAN64_OUT=... # Output directory
./chainer_dcgan.py \
  --arch dcgan64 \
  --image_size 64 \
  --adam_alpha 0.0001 --adam_beta1 0.5 --adam_beta2 0.999 --lambda_gp 1.0 --learning_rate_anneal 0.9 --learning_rate_anneal_trigger 0 --learning_rate_anneal_interval 5000 --max_iter 100000 --snapshot_interval 5000 --evaluation_sample_interval 100 --display_interval 10 \
  --npz_path $DATA_FILE_SIZE_64 \
  --out $DCGAN64_OUT \
  ;


# Training ReSNet128 model
DATA_FILE_SIZE_128=...  # Data file of size 64
RESNET128_OUT=... # Output directory
./chainer_dcgan.py \
  --arch dcgan64 \
  --image_size 64 \
  --adam_alpha 0.0001 --adam_beta1 0.5 --adam_beta2 0.999 --lambda_gp 1.0 --learning_rate_anneal 0.9 --learning_rate_anneal_trigger 0 --learning_rate_anneal_interval 5000 --max_iter 100000 --snapshot_interval 5000 --evaluation_sample_interval 100 --display_interval 10 \
  --npz_path $DATA_FILE_SIZE_128 \
  --out $RESNET128_OUT \
  ;


# Training ReSNet256 model
DATA_FILE_SIZE_256=...  # Data file of size 64
RESNET256_OUT=... # Output directory
./chainer_dcgan.py \
  --arch dcgan64 \
  --image_size 64 \
  --adam_alpha 0.0001 --adam_beta1 0.5 --adam_beta2 0.999 --lambda_gp 1.0 --learning_rate_anneal 0.9 --learning_rate_anneal_trigger 0 --learning_rate_anneal_interval 5000 --max_iter 100000 --snapshot_interval 5000 --evaluation_sample_interval 100 --display_interval 10 \
  --npz_path $DATA_FILE_SIZE_256 \
  --out $RESNET256_OUT \
  ;

```

### Step 3 - Convert from Chainer model to Keras/Tensorflow.js model

Note that due to difficulty in training GANs,
you may want to select a proper snapshot by specifying `ITER` below.
This script also samples a few images serving as a sanity check and providing clue for picking the correct snapshot.

```
# DCGAN64

ITER=50000
./dcgan_chainer_to_keras.py \
  --arch dcgan64 \
  --chainer_model_path $DCGAN64_OUT/SmoothedGenerator_${ITER}.npz \
  --keras_model_path $DCGAN64_OUT/Keras_SmoothedGenerator_${ITER}.h5 \
  --tfjs_model_path $DCGAN64_OUT/tfjs_SmoothedGenerator_${ITER} \
  ;

# ResNet128

ITER=20000
./dcgan_chainer_to_keras.py \
  --arch resnet128 \
  --chainer_model_path $RESNET128_OUT/SmoothedGenerator_${ITER}.npz \
  --keras_model_path $RESNET128_OUT/Keras_SmoothedGenerator_${ITER}.npz.h5 \
  --tfjs_model_path $RESNET128_OUT/tfjs_SmoothedGenerator_${ITER} \
  ;

# ResNet256

ITER=45000
./dcgan_chainer_to_keras.py \
  --arch resnet256 \
  --chainer_model_path $RESNET256_OUT/SmoothedGenerator_${ITER}.npz \
  --keras_model_path $RESNET256_OUT/Keras_SmoothedGenerator_${ITER}.npz.h5 \
  --tfjs_model_path $RESNET256_OUT/tfjs_SmoothedGenerator_${ITER} \
  ;

```

### Step 4 - Present the generative as a web page

This step is covered by a web project under `./webcode/ganshowcase`.
Now it is assumeed that you are in `./webcode/ganshowcase` directory.

First you need to copy TensorFlow.js model (specified as argument to `--tfjs_model_path` in previous step) to a public accessible place, and
modify `model_url` in `all_model_info` which is in the beginning of `index.js`.

Then run the following:
```
yarn
yarn build
```

Finally, copy `./dist/`, which is the built web page and js file,
to whatever suitable for web hosting.
