# Text binarization with tensorflow

### Requirements
Tested with tensorflow 1.6. Recommended installation via virtualenv:
```sh
# Clone repo
git clone --recursive git@gitlab.cs.fau.de:ko01jaxu/ma-proj.git

# Setup virtualenv for crf compilation and training
TF_VERSION=1.6

# Change MY_ENV_DIR to the desired virtualenv directory.
MY_ENV_DIR=/tmp/tf-$TF_VERSION-env
virtualenv --system-site-packages -p python3 $MY_ENV_DIR

# If you have virtualenvwrapper installed use bin/postactivate for this
echo "export CUDA_HOME=/usr/local/cuda-9.0/" >> $MY_ENV_DIR/bin/activate
echo "export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64" >> $MY_ENV_DIR/bin/activate
source $MY_ENV_DIR/bin/activate

easy_install -U pip
pip3 install --upgrade pip
pip3 install --upgrade tensorflow-gpu==$TF_VERSION Pillow opencv-python

# Compile CRF extension
cd ma-proj/
./utils/crfasrnn/compile_high_dim_filter.sh
```

### Run
```sh
python train.py --helpusage: train.py [-h] [--data_dir DATA_DIR] [--model_dir MODEL_DIR]
                [--unet_depth UNET_DEPTH] [--filter_size FILTER_SIZE]
                [--train_epochs TRAIN_EPOCHS]
                [--epochs_per_eval EPOCHS_PER_EVAL] [--batch_size BATCH_SIZE]
                [--buffer_size BUFFER_SIZE] [--img_patch_size IMG_PATCH_SIZE]
                [--scale_factor SCALE_FACTOR] [--dataset DATASET]
                [--crf_training CRF_TRAINING] [--only_crf ONLY_CRF]
                [--transfer TRANSFER]

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   The path to the dataset directory.
  --model_dir MODEL_DIR
                        The directory where the model will be stored.
  --unet_depth UNET_DEPTH
                        The size of the Unet model to use.
  --filter_size FILTER_SIZE
                        Convolution filter size.
  --train_epochs TRAIN_EPOCHS
                        The number of epochs to train.
  --epochs_per_eval EPOCHS_PER_EVAL
                        The number of batches to run in between evaluations.
  --batch_size BATCH_SIZE
                        The number of images per batch.
  --buffer_size BUFFER_SIZE
                        The number of images to buffer for training.
  --img_patch_size IMG_PATCH_SIZE
                        Input image size using random crop and pad. If not
                        provided use a dataset specific default.
  --scale_factor SCALE_FACTOR
                        Input image scale factor between (0, 1].
  --dataset DATASET     The dataset to train with. Possible datasets: `dibco`,
                        `hisdb`.
  --crf_training CRF_TRAINING
                        After normal training train a downstream crf
  --only_crf ONLY_CRF   Start immediately with crf training
  --transfer TRANSFER   Use with pre-trained checkpoint file;Only the
                        downscale conv layers will get restored.
```
Example:
```sh
# Make sure the virtualenv is active

DATA_SET=dibco
MODEL_DIR=/tmp/models/$DATA_SET/crf_d5_fs7
DATA_DIR=/tmp/$DATA_SET

python3 train.py \
    --unet_depth=5 \
    --filter_size=7 \
    --model_dir=$MODEL_DIR \
    --data_dir=$DATA_DIR \
    --dataset=$DATA_SET \
    --batch_size=4 \
    --epochs_per_eval=200 \
    --train_epochs=5000 \
    --buffer_size=200 \
    --crf_training=True

python3 predict.py \
    --unet_depth=5 \
    --filter_size=7 \
    --model_dir=$MODEL_DIR \
    --data_dir=$DATA_DIR \
    --dataset=$DATA_SET \
    $DATA_DIR/*
```
