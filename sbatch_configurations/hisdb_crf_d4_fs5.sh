#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=12000
#SBATCH --gres=gpu:1
#SBATCH -o /cluster/%u/%j.out 
#SBATCH -e /cluster/%u/%j.out

TF_VERSION=1.4.0

export CUDA_HOME=/cluster/ko01jaxu/cuda-8.0
export LD_LIBRARY_PATH=/cluster/ko01jaxu/cuda-8.0/lib64

mkdir -p /scratch/ko01jaxu/env/
virtualenv --system-site-packages -p python3 /scratch/ko01jaxu/env/tf-$TF_VERSION
source /scratch/ko01jaxu/env/tf-$TF_VERSION/bin/activate
easy_install -U pip
pip3 install --upgrade tensorflow-gpu==$TF_VERSION Pillow

cd /cluster/ko01jaxu/ma-proj
./utils/crfasrnn/compile_high_dim_filter.sh

DATA_SET=hisdb
MODEL_DIR=/cluster/ko01jaxu/models/$DATA_SET/crf_d4_fs5
DATA_DIR=/tmp/ko01jaxu/$DATA_SET
if [ -d "$DATA_DIR" ]; then
  rm -rf $DATA_DIR
fi

python3 /cluster/ko01jaxu/ma-proj/train.py \
    --unet_depth=4 \
    --filter_size=5 \
    --img_patch_size=250 \
    --model_dir=$MODEL_DIR \
    --data_dir=$DATA_DIR \
    --batch_size=8 \
    --epochs_per_eval=500 \
    --train_epochs=1000 \
    --buffer_size=50 \
    --dataset=$DATA_SET \
    --crf_training=True

deactivate

rm -rf /scratch/ko01jaxu
