#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8096
#SBATCH --gres=gpu:1
#SBATCH -o /cluster/%u/%j.out 
#SBATCH -e /cluster/%u/%j.out

printf 'Using preinstalled tensorflow:\n'
python3 -c 'import tensorflow as tf; print(tf.__version__)'

printf 'Using virtualenv with tensorflow 1.4:\n'
mkdir -p /scratch/ko01jaxu/env/
export CUDA_HOME=/cluster/ko01jaxu/cuda-8.0
export LD_LIBRARY_PATH=/cluster/ko01jaxu/cuda-8.0/lib64
virtualenv --system-site-packages -p python3 /scratch/ko01jaxu/env/tf-1.4
source /scratch/ko01jaxu/env/tf-1.4/bin/activate
easy_install -U pip
pip3 install --upgrade 'tensorflow-gpu==1.4.0' Pillow
python3 -c 'import tensorflow as tf; print(tf.__version__)'
deactivate

rm -rf /scratch/ko01jaxu
