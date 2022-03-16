# Speech Emotion Recognition

### Prerequisite
use the following command to install packages:
'''sh
pip install mlflow
pip install ujson
pip install transformers
'''
The torch version is `1.10.0+cu111`.

### Extract Features 

#### 1. Allosaurus Features
The `run.py` file in the original Allosaurus project is modified in order to extract the feature embedding just before the softmax layer. In order to get the Allosaurus embedding for your wav files, please change the `TRAIN_PATH` in `run.py` as the csv files containing all the training samples, and change `ALLO_EMB_PATH` to the directory in which you want to store the feature pkl.
Then simply run
```sh
python -m allosaurus.run
```
to extract allosaurus features. 

### Train the Model
Firstly, you need to change the paths in file `allosaurus+CNN/SER/training/utils.py` to the path in your own environment.

#### 1. English Dataset

`train.sh` is the command to train IEMOCAP dataset, which has the command:
```sh
CUDA_VISIBLE_DEVICES="0" \
python -m \
SER.train --bsize 512 --accum 1 --lr 0.001 --GE2E --MFCC \
--triples /content/drive/MyDrive/Speech-Emotion-Recognition/allosaurus+CNN/iemocap/_iemocap_03F.train.csv \
--langs "en" \
--root /content/drive/MyDrive/Speech-Emotion-Recognition/allosaurus+CNN/ --experiment MSMARCO-psg --run msmarco.psg.rronly --maxsteps 20000
```

In order to train the model, please firstly change the `--triples` as the csv files containing audio file names and labels; change `--root` as the path towards `allosaurus+CNN` in your environment. Then run the following command:

```sh
 cd allosaurus+CNN
 sh train.sh
```

#### 2. German Dataset
Similarly, train the German EMODB dataset with command:
```sh
sh train_ge.sh
```

#### 3. Persian Dataset
Train the Persian dataset with command:
```sh
sh train_pe.sh
```
