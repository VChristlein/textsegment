#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus_per-task=1
#SBATCH --mem=8096
#SBATCH --gres=gpu:1
#SBATCH -o /home/%u/%j.out 
#SBATCH -e /home/%u/%j.err 

pyhton3 -c 'import tensorflow as tf; print(tf.__version__)'

mkdir /scratch/%u/env/tf-1.4
virtualenv --system-site-packages -p python3 /scratch/%u/env/tf-1.4
source activate /scratch/%u/env/tf-1.4
easy_install -U pip
pip3 install --upgrade tensorflow-gpu
pyhton3 -c 'import tensorflow as tf; print(tf.__version__)'

rm -rf /scratch/%u/env/tf-1.4
