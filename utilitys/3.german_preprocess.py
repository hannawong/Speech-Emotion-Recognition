import numpy as np 
import pandas as pd 

import os

def decompose_emodb():
    EMODB_PATH = '/data1/jiayu_xiao/project/wzh/data/ger_wav'
    emotion = []
    path = []
    gender = []

    for root, dirs, files in os.walk(EMODB_PATH):
        for name in files:
            if name[0:2] in '0310111215':  # MALE
                gender.append("M")
                if name[5] == 'W':  # Ärger (Wut) -> Angry
                    emotion.append('angry')
                elif name[5] == 'L':  # Langeweile -> Boredom
                    emotion.append('bored')
                elif name[5] == 'E':  # Ekel -> Disgusted
                    emotion.append('disgust')
                elif name[5] == 'A':  # Angst -> Angry
                    emotion.append('fear')
                elif name[5] == 'F':  # Freude -> Happiness
                    emotion.append('happy')
                elif name[5] == 'T':  # Trauer -> Sadness
                    emotion.append('sad')
                elif name[6] == 'N':
                    emotion.append('neutral')
                else:
                    emotion.append('unknown')
            else:
                gender.append("F")
                if name[5] == 'W':  # Ärger (Wut) -> Angry
                    emotion.append('angry')
                elif name[5] == 'L':  # Langeweile -> Boredom
                    emotion.append('bored')
                elif name[5] == 'E':  # Ekel -> Disgusted
                    emotion.append('disgust')
                elif name[5] == 'A':  # Angst -> Angry
                    emotion.append('fear')
                elif name[5] == 'F':  # Freude -> Happiness
                    emotion.append('happy')
                elif name[5] == 'T':  # Trauer -> Sadness
                    emotion.append('sad')
                elif name[6] == 'N':
                    emotion.append('neutral')
                else:
                    emotion.append('unknown')

            path.append(os.path.join(EMODB_PATH, name))
    print(len(emotion),len(path),len(gender))
    emodb_df = pd.DataFrame({"label":emotion,"path":path,"gender":gender})
    
    return emodb_df

df = decompose_emodb()
print(df)
PATH = "/data1/jiayu_xiao/project/wzh_recommendation/Speech-Emotion-Recognition/allosaurus+CNN/emodb/"
df = df[df["label"]!="unknown"]
df.to_csv(PATH+"total.csv",index = False)