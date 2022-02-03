# Speech Emotion Recognition

### Extract Features 

#### 1. Allosaurus Features
The `run.py` file in the original Allosaurus project is modified in order to extract the feature embedding just before the softmax layer. In order to get the Allosaurus embedding for your wav files, please change the `TRAIN_PATH` in `run.py` as the csv files containing all the training samples, and change `ALLO_EMB_PATH` to the directory in which you want to store the feature pkl.


### Train the Model

#### 1. English Dataset
In order to train the model, please firstly change the `--triples` as the csv files containing audio file names and labels; change `--root` as the path towards `allosaurus+CNN` in your environment. Then run the following command:

```sh
 cd allosaurus+CNN
 sh train.sh
```
