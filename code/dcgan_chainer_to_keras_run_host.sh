#!/usr/bin/env bash

set -e

function deal() {
  arch=$1
  dir=$2

  iters=`ls ${dir}/SmoothedGenerator_*.npz | grep -o '[[:digit:]]*\.npz' | sed -e 's/\.npz//'`
  iters_array=($iters)

  for iter in "${iters_array[@]}" ; do

    if [ -f "${dir}/Keras_SmoothedGenerator_${iter}.npz.h5" ] ; then
      continue
    fi

    CUDA_VISIBLE_DEVICES=""  \
    ../scripts/run_docker.sh \
    ./dcgan_chainer_to_keras.py \
      --arch $arch \
      --chainer_model_path ${dir}/SmoothedGenerator_${iter}.npz \
      --keras_model_path ${dir}/Keras_SmoothedGenerator_${iter}.npz.h5 \
      --tfjs_model_path ${dir}/tfjs_SmoothedGenerator_${iter} \
      ;

    scp -r \
      ${dir}/tfjs_SmoothedGenerator_${iter} \
      lab-ws:~/nginx/public/share/tfjs_gan/`basename "${dir}"` \
      ;
  done
}

deal 'resnet256' '../scratch/model/chainer-resent256-celebahq-256'
deal 'resnet128'  '../scratch/model/chainer-resent128-celebahq-128'
deal 'dcgan64' '../scratch/model/chainer-dcgan-celebahq-64/'
