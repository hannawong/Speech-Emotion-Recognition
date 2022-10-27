# MULTILINGUAL SPEECH EMOTION RECOGNITION WITH MULTI-GATING MECHANISM AND NEURAL ARCHITECTURE SEARCH

### Prerequisite

#### 1. Install packages
use the following command to install packages:

```sh
pip install -r requirements.txt
```
The cuda version is `11.1`.

#### 2. Download Dataset and Features

Wav files: https://drive.google.com/drive/folders/12V4A78u_XOp7vXlUrF5vweXEF9nb4btJ?usp=sharing

Allosaurus features: https://drive.google.com/drive/folders/1YDjsl05ubArpvo4kk9yzXjnbexXgInHP?usp=sharing

GE2E features is in data/GE2E

MFCC features: https://drive.google.com/drive/folders/12aaC-n4o0qXSL53rRFcLWqx6ar1J4iUs?usp=sharing


### Train the Model
Firstly, you need to change the paths in file `model/SER_mmoe/training/utils.py` and `model/SER_mmoe/modeling/ser_model.py`to the path in your own environment.

The training process is very simple. Please run the following command:

```sh
sh train.sh
```