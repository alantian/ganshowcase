
# data processing

```
../scripts/run_docker.sh \
./datatool.py --task dir_to_npz \
  --dir_path ../scratch/data/celeba/dir/img_align_celeba_png \
  --npz_path ../scratch/data/celeba/npz/img_align_celeba_png-size=64.npz \
  --size 64

../scripts/run_docker.sh \
./datatool.py --task npz_to_dir \
  --npz_path ../scratch/data/celeba/npz/img_align_celeba_png-size=64.npz \
  --dir_path ../scratch/data/celeba/dir/img_align_celeba_png-size=64 \
  --size 64  \
  --max_images 10



../scripts/run_docker.sh \
./datatool.py --task multisize_h5_to_npz \
  --multisize_h5_path ../scratch2/data/celebahq/multisize_h5/celeba_hq.h5 \
  --npz_path ../scratch/data/celebahq/npz/celeba_hq-size=64.npz \
  --size 64


../scripts/run_docker.sh \
./datatool.py --task npz_to_dir \
  --npz_path ../scratch/data/celebahq/npz/celeba_hq-size=64.npz \
  --dir_path ../scratch/data/celebahq/dir/celeba_hq-size=64 \
  --size 64 \
  --max_images 10


../scripts/run_docker.sh \
./datatool.py --task multisize_h5_to_npz \
  --multisize_h5_path ../scratch2/data/celebahq/multisize_h5/celeba_hq.h5 \
  --npz_path ../scratch/data/celebahq/npz/celeba_hq-size=128.npz \
  --size 128


../scripts/run_docker.sh \
./datatool.py --task npz_to_dir \
  --npz_path ../scratch/data/celebahq/npz/celeba_hq-size=128.npz \
  --dir_path ../scratch/data/celebahq/dir/celeba_hq-size=128 \
  --size 128 \
  --max_images 10


../scripts/run_docker.sh \
./datatool.py --task multisize_h5_to_npz \
  --multisize_h5_path ../scratch2/data/celebahq/multisize_h5/celeba_hq.h5 \
  --npz_path ../scratch/data/celebahq/npz/celeba_hq-size=256.npz \
  --size 256


../scripts/run_docker.sh \
./datatool.py --task npz_to_dir \
  --npz_path ../scratch/data/celebahq/npz/celeba_hq-size=256.npz \
  --dir_path ../scratch/data/celebahq/dir/celeba_hq-size=256 \
  --size 256 \
  --max_images 10


```


# training

```

CUDA_VISIBLE_DEVICES="0" \
../scripts/run_docker.sh \
./keras_dcgan.py \
  --npz_path ../scratch/data/celebahq/npz/celeba_hq-size=64.npz \
  --out ../scratch/model/keras-dcgan-celebahq-64 \
  --size 64 \
  --gptype DRAGAN \
  --weight_scale 0.05 \
  ;

CUDA_VISIBLE_DEVICES="0" \
../scripts/run_docker.sh \
./chainer_dcgan.py \
  --arch dcgan64 \
  --image_size 64 \
  --adam_alpha 0.0001 --adam_beta1 0.5 --adam_beta2 0.999 --lambda_gp 1.0 --learning_rate_anneal 0.9 --learning_rate_anneal_trigger 0 --learning_rate_anneal_interval 5000 --max_iter 100000 --snapshot_interval 5000 --evaluation_sample_interval 100 --display_interval 10 \
  --npz_path ../scratch/data/celebahq/npz/celeba_hq-size=64.npz \
  --out ../scratch/model/chainer-dcgan-celebahq-64 \
  ;



CUDA_VISIBLE_DEVICES="0" \
../scripts/run_docker.sh \
./chainer_dcgan.py \
  --arch resnet128 \
  --image_size 128 \
  --batch_size 32 \
  --adam_alpha 0.0001 --adam_beta1 0.5 --adam_beta2 0.999 --lambda_gp 1.0 --learning_rate_anneal 0.9 --learning_rate_anneal_trigger 0 --learning_rate_anneal_interval 5000 --max_iter 100000 --snapshot_interval 5000 --evaluation_sample_interval 100 --display_interval 10 \
  --npz_path ../scratch/data/celebahq/npz/celeba_hq-size=128.npz \
  --out ../scratch/model/chainer-resent128-celebahq-128 \
  ;


CUDA_VISIBLE_DEVICES="1" \
../scripts/run_docker.sh \
./chainer_dcgan.py \
  --arch resnet256 \
  --image_size 256 \
  --batch_size 32 \
  --adam_alpha 0.0001 --adam_beta1 0.5 --adam_beta2 0.999 --lambda_gp 1.0 --learning_rate_anneal 0.9 --learning_rate_anneal_trigger 0 --learning_rate_anneal_interval 5000 --max_iter 100000 --snapshot_interval 5000 --evaluation_sample_interval 100 --display_interval 10 \
  --npz_path ../scratch/data/celebahq/npz/celeba_hq-size=256.npz \
  --out ../scratch/model/chainer-resent256-celebahq-256 \
  ;




```


# Convert from chainer to keras / tfjs

```

####### dcgan64

CUDA_VISIBLE_DEVICES="2" \
../scripts/run_docker.sh \
./dcgan_chainer_to_keras.py \
  --arch dcgan64 \
  --chainer_model_path  ../scratch/model/chainer-dcgan-celebahq-64/SmoothedGenerator_50000.npz \
  --keras_model_path  ../scratch/model/chainer-dcgan-celebahq-64/Keras_SmoothedGenerator_50000.h5 \
  --tfjs_model_path ../scratch/model/chainer-dcgan-celebahq-64/tfjs_SmoothedGenerator_50000 \
  ;


scp -r \
  ../scratch/model/chainer-dcgan-celebahq-64/tfjs_SmoothedGenerator_50000 \
  lab-ws:~/nginx/public/share/tfjs_gan/chainer-dcgan-celebahq-64 \
  ;




######## resnet128
CUDA_VISIBLE_DEVICES="2" \
../scripts/run_docker.sh \
./dcgan_chainer_to_keras.py \
  --arch resnet128 \
  --chainer_model_path  ../scratch/model/chainer-resent128-celebahq-128/SmoothedGenerator_20000.npz \
  --keras_model_path  ../scratch/model/chainer-resent128-celebahq-128/Keras_SmoothedGenerator_20000.npz.h5 \
  --tfjs_model_path ../scratch/model/chainer-resent128-celebahq-128/tfjs_SmoothedGenerator_20000 \
  ;

scp -r \
  ../scratch/model/chainer-resent128-celebahq-128/tfjs_SmoothedGenerator_20000 \
  lab-ws:~/nginx/public/share/tfjs_gan/chainer-resent128-celebahq-128 \
  ;



########## resnet256

ITER=40000

CUDA_VISIBLE_DEVICES=""  \
../scripts/run_docker.sh \
./dcgan_chainer_to_keras.py \
  --arch resnet256 \
  --chainer_model_path  ../scratch/model/chainer-resent256-celebahq-256/SmoothedGenerator_${ITER}.npz \
  --keras_model_path  ../scratch/model/chainer-resent256-celebahq-256/Keras_SmoothedGenerator_${ITER}.npz.h5 \
  --tfjs_model_path ../scratch/model/chainer-resent256-celebahq-256/tfjs_SmoothedGenerator_${ITER} \
  ;


scp -r \
  ../scratch/model/chainer-resent256-celebahq-256/tfjs_SmoothedGenerator_${ITER} \
  lab-ws:~/nginx/public/share/tfjs_gan/chainer-resent256-celebahq-256 \
  ;


```


# Sync to Google Cloud

_IMPORTANT: Run on lab-ws_

```
gsutil -m rsync -R ~/nginx/public/share/tfjs_gan/chainer-dcgan-celebahq-64/tfjs_SmoothedGenerator_50000 gs://store.alantian.net/tfjs_gan/chainer-dcgan-celebahq-64/tfjs_SmoothedGenerator_50000
gsutil -m rsync -R ~/nginx/public/share/tfjs_gan/chainer-resent128-celebahq-128/tfjs_SmoothedGenerator_20000 gs://store.alantian.net/tfjs_gan/chainer-resent128-celebahq-128/tfjs_SmoothedGenerator_20000
gsutil -m rsync -R ~/nginx/public/share/tfjs_gan/chainer-resent256-celebahq-256/tfjs_SmoothedGenerator_40000 gs://store.alantian.net/tfjs_gan/chainer-resent256-celebahq-256/tfjs_SmoothedGenerator_40000


gsutil iam ch allUsers:objectViewer gs://store.alantian.net


# set cache
# 3600 = 1 hour / 604800 = 7 days
gsutil -m setmeta \
  -h "Cache-Control:public, max-age=604800" \
  gs://store.alantian.net/tfjs_gan/*/*/*
```
