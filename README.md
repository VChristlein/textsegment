# Text binarization with tensorflow on the [DIBCO challenges][1]

This work was done by Hendrik Schröter in the course of a master's project (under my supervision).

### Results

I trained the model on the DIBCO dataset (2009-2016). This are the
(validation) results for the 2017 challenge:

| Type | F-Measure | Pseudo F-Measure | PSNR   | DRD   |
|------|-----------|------------------|--------|-------|
| Both | 90,048    | 93,9915          | 17,561 | 3,064 |
| H    | 89,106	   | 94,004           | 16,916 | 3,331 |
| P    | 90,99	   | 93,979           | 18,206 | 2,797 |


Individual image results:

|    | Type | F-Measure | Pseudo F-Measure | PSNR  | DRD  |
|----|------|-----------|------------------|-------|------|
| 1	 | H    | 86,56     | 93,23            | 16,29 | 3,48 |
| 2	 | H    | 88,08     | 93,56            | 16,48 | 3,92 |
| 3	 | H    | 88,46     | 95,25            | 17,76 | 3,16 |
| 4	 | H    | 86,57     | 92,88            | 18,15 | 4,02 |
| 5	 | H    | 87,90     | 92,54            | 20,44 | 3,73 |
| 6	 | H    | 93,24     | 95,55            | 15,23 | 2,42 |
| 7	 | H    | 92,64     | 94,08            | 15,07 | 2,77 |
| 8	 | H    | 88,53     | 95,08            | 17,64 | 3,88 |
| 9	 | H    | 88,19     | 92,19            | 16,10 | 3,32 |
| 10 | H    | 90,89     | 95,68            | 16,00 | 2,61 |
| 11 | P    | 94,48     | 93,99            | 18,04 | 2,66 |
| 12 | P    | 89,56     | 92,05            | 17,10 | 3,46 |
| 13 | P    | 89,97     | 89,71            | 18,23 | 3,62 |
| 14 | P    | 91,68     | 94,71            | 19,01 | 2,43 |
| 15 | P    | 93,20     | 94,56            | 16,85 | 1,97 |
| 16 | P    | 93,79     | 94,46            | 21,60 | 2,01 |
| 17 | P    | 89,85     | 95,89            | 19,99 | 2,27 |
| 18 | P    | 87,86     | 95,26            | 16,53 | 3,60 |
| 19 | P    | 89,43     | 91,41            | 18,00 | 3,38 |
| 20 | P    | 90,08     | 97,75            | 16,71 | 2,57 |

H: Handwritten
P: Printed

#### Links

Pretrained model from the results above:
https://www.dropbox.com/sh/9r9ep5ee95k22xk/AABr0gDllCX7xXD2LE9Ex6bwa?dl=0

Predicted ground truth:
https://www.dropbox.com/sh/tql95946q7uwkye/AABq-v7YFSTaoPlaDPypSFyUa?dl=0


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
    --unet_depth=4 \
    --filter_size=5 \
    --model_dir=$MODEL_DIR \
    --dataset=$DATA_SET \
    $DATA_DIR/*
```

[1]: https://vc.ee.duth.gr/h-dibco2018/

#### Acknowledgement

Development for this tool has been supported by the European project 211 "Modern access to historical sources" a Cross-border cooperation program Free State of Bavaria and the Czech Republic.

