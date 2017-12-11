#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12000
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH -o /cluster/%u/%j.out 
#SBATCH -e /cluster/%u/%j.err 

printf 'Using virtualenv:\n'
mkdir /scratch/ko01jaxu/env/
virtualenv --system-site-packages -p python3 /scratch/ko01jaxu/env/tf-1.2
source /scratch/ko01jaxu/env/tf-1.2/bin/activate
easy_install -U pip
pip3 install --upgrade 'tensorflow-gpu==1.2.0' Pillow

python3 /cluster/ko01jaxu/ma-proj/train.py \
    --model_dir=/cluster/ko01jaxu/unet_model \
#    --batch_size=8 \  # Slow on cudnn 5
    --epochs_per_eval=100 \
    --train_epochs=10000 \
    --buffer_size=1000 

deactivate

rm -rf /scratch/ko01jaxu
