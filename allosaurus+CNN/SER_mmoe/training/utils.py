import os
import torch
import random
import pickle as pkl
import numpy as np
import pandas as pd
import librosa
from SER.utils.runs import Run
from SER.utils.utils import save_checkpoint
from SER.parameters import SAVED_CHECKPOINTS

MAX_LEN = 250
WAV_PATH = "/data/jiayu_xiao/IEMOCAP/path_to_wavs/"
ALLO_EMB_PATH = "/data/jiayu_xiao/IEMOCAP/allo_embedding/"
GE2E_EMB_PATH = "../data/GE2E/"
MFCC_EMB_PATH = "/data/jiayu_xiao/IEMOCAP/mfcc_embeddings/"
WAV2VEC1_EMB_PATH = "/data/jiayu_xiao/IEMOCAP/wav2vec1_embeddings/"
BLOY_EMB_PATH = "/data/jiayu_xiao/IEMOCAP/byols_embeddings/"


def print_progress(scores):
    positive_avg, negative_avg = round(scores[:, 0].mean().item(), 2), round(scores[:, 1].mean().item(), 2)
    print("#>>>   ", positive_avg, negative_avg, '\t\t|\t\t', positive_avg - negative_avg)


def manage_checkpoints(args, model, optimizer, batch_idx):
    arguments = args.input_arguments.__dict__

    path = os.path.join(Run.path, 'checkpoints')

    if not os.path.exists(path):
        os.mkdir(path)

    if batch_idx % args.save_step == 0:
        name = os.path.join(path, "model.dnn")
        save_checkpoint(name, 0, batch_idx, model, optimizer, arguments)

    if batch_idx in SAVED_CHECKPOINTS:
        name = os.path.join(path, "model-{}.dnn".format(batch_idx))
        save_checkpoint(name, 0, batch_idx, model, optimizer, arguments)

def shuf_order(langs):
    """
    Randomize training order.
    """
    tmp = langs
    random.shuffle(tmp)
    return tmp


def tensorize_triples(args,lang,audio_files, labels, bsize): ##transform sentence into ids and masks
    #assert bsize is None or len(audio_files) % bsize == 0
    print(lang)
    if lang == "fr":
        MAX_LEN_MFCC = 2000
    else:
        MAX_LEN_MFCC = 250
    allo_embs = []
    length = []
    for file in audio_files:
        allo_emb = pkl.load(open(ALLO_EMB_PATH+file.split("/")[-1]+".pkl","rb"))
        allo_emb = allo_emb.squeeze()

        if allo_emb.shape[0] >= MAX_LEN:
            allo_emb = allo_emb[:MAX_LEN,:]
            length.append(MAX_LEN-1)
        else:
            length.append(allo_emb.shape[0]-1)
            zero = torch.zeros((MAX_LEN-allo_emb.shape[0], allo_emb.shape[1]))
            allo_emb = torch.cat((allo_emb, zero), dim=0)
        if torch.isnan(allo_emb).any():
            allo_emb = torch.zeros((MAX_LEN,allo_emb.shape[1]))
        allo_embs.append(allo_emb)
    allosaurus_embedding = torch.stack(allo_embs,axis = 0) ##len,emb_size


    GE2E_embs = []
    bloy_embs = []
    WAV2VEC1_embs = []
    wav2vec1_length = []
    for file in audio_files:
        ge2e_emb = np.load(open(GE2E_EMB_PATH+file.split("/")[-1][:-4]+".npy","rb"))
        ge2e_emb = np.expand_dims(ge2e_emb,0)
        ge2e_emb = torch.Tensor(ge2e_emb)
        GE2E_embs.append(ge2e_emb)

        b_emb = torch.load(BLOY_EMB_PATH+file.split("/")[-1][:-4]+".pt")
        b_emb = np.expand_dims(b_emb,0)
        b_emb = torch.Tensor(b_emb)
        bloy_embs.append(b_emb)


        try:
            wav2vec1_emb = torch.load(WAV2VEC1_EMB_PATH+file.split("/")[-1][:-4]+".pt")#(1,512,xxxdepend on length)
        except:
            wav2vec1_emb = torch.zeros((1,512,250))

        wav2vec1_emb = wav2vec1_emb.permute(0,2,1)#exchange dim1 and dim2
        wav2vec1_emb = wav2vec1_emb.squeeze()#(num_of_window,512)

        
        if wav2vec1_emb.shape[0] >= MAX_LEN:
            wav2vec1_emb = wav2vec1_emb[:MAX_LEN,:]
            wav2vec1_length.append(MAX_LEN-1)
        else:
            wav2vec1_length.append(wav2vec1_emb.shape[0]-1)
            zero = torch.zeros((MAX_LEN - wav2vec1_emb.shape[0],wav2vec1_emb.shape[1]))
            wav2vec1_emb = torch.cat((wav2vec1_emb,zero),dim = 0)

        WAV2VEC1_embs.append(wav2vec1_emb)

    GE2E_embedding = torch.stack(GE2E_embs,axis = 0)
    bloy_embedding = torch.stack(bloy_embs,axis = 0)
    WAV2VEC1_embedding = torch.stack(WAV2VEC1_embs,axis = 0)



    

    mfcc_embs = []
    mfcc_length = []

    for file in audio_files:
        if lang == "en":
            mfccs = pkl.load(open(MFCC_EMB_PATH+file.split("/")[-1]+".pkl","rb"))
        else:
            mfccs = np.load(MFCC_EMB_PATH+file.split("/")[-1]+".npy")
        mfccs = torch.Tensor(np.transpose(mfccs))
        mfcc_length.append(min(mfccs.shape[0],MAX_LEN_MFCC-1))
        if mfccs.shape[0] >= MAX_LEN_MFCC:
            mfcc_emb = mfccs[:MAX_LEN_MFCC,:]
        else:
            zero = torch.zeros((MAX_LEN_MFCC-mfccs.shape[0], mfccs.shape[1]))
            mfcc_emb = torch.cat((mfccs, zero), dim=0)
            
        mfcc_embs.append(mfcc_emb)
    mfcc_embedding = torch.stack(mfcc_embs,axis = 0) ##len,emb_size
    #else:
        #mfcc_embedding = torch.zeros((1,MAX_LEN,24))################此处存疑################
    
    query_batches = _split_into_batches(allosaurus_embedding,GE2E_embedding,WAV2VEC1_embedding,mfcc_embedding,labels,length,wav2vec1_length,mfcc_length,bloy_embedding,audio_files,bsize)
    batches = []
    for (embed,GE2E, WAV2VEC1,MFCC,label,length,wav2vec1_length,mfcc_length,bloy_embedding,audio_files) in query_batches:
        Q = (embed,GE2E,WAV2VEC1,MFCC,label,length,wav2vec1_length,mfcc_length,bloy_embedding,audio_files)
        batches.append(Q)
    return batches


def _split_into_batches(allosaurus_embedding, GE2E_embedding,WAV2VEC1_embedding,mfcc_embedding, labels,length,wav2vec1_length,mfcc_length,bloy_embedding,audio_files,bsize):
    batches = []
    for offset in range(0, allosaurus_embedding.shape[0], bsize):
        batches.append((allosaurus_embedding[offset:offset+bsize], GE2E_embedding[offset:offset+bsize],WAV2VEC1_embedding[offset:offset+bsize],mfcc_embedding[offset:offset+bsize],labels[offset:offset+bsize],length[offset:offset+bsize],wav2vec1_length[offset:offset+bsize],mfcc_length[offset:offset+bsize],bloy_embedding[offset:offset+bsize],audio_files[offset:offset+bsize]))
    return batches


from sklearn.utils import shuffle
def split_train_val_test_german(path):
    df = pd.read_csv(path)
    df = shuffle(df)
    df_train = df[:int(len(df)*0.7)]
    df_val = df[int(len(df)*0.7):int(len(df)*0.8)]
    df_test = df[int(len(df)*0.8):]
    df_train.to_csv(path[:-9]+"train.csv")
    df_test.to_csv(path[:-9]+"test.csv")
    df_val.to_csv(path[:-9]+"val.csv")
    return path[:-9]+"train.csv",path[:-9]+"test.csv",path[:-9]+"val.csv"