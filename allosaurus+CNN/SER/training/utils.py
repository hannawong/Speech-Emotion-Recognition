import os
import torch
import random
import pickle as pkl
import numpy as np
from SER.utils.runs import Run
from SER.utils.utils import save_checkpoint
from SER.parameters import SAVED_CHECKPOINTS

MAX_LEN = 100


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


def tensorize_triples(audio_files, labels, bsize): ##transform sentence into ids and masks
    assert bsize is None or len(audio_files) % bsize == 0
    allo_embs = []
    for file in audio_files:
        allo_emb = pkl.load(open("/data/jiayu_xiao/IEMOCAP/allo_embedding/"+file.split("/")[-1]+".pkl","rb"))
        if allo_emb.shape[1] >= 100:
            allo_emb = allo_emb[0,:MAX_LEN,:]
        else:
            zero = torch.zeros((MAX_LEN-allo_emb.shape[1], allo_emb.shape[2]))
            allo_emb = torch.cat((allo_emb[0], zero), dim=0)
        allo_embs.append(allo_emb)
    allosaurus_embedding = torch.stack(allo_embs,axis = 0) ##len,emb_size
    query_batches = _split_into_batches(allosaurus_embedding,labels,bsize)
    batches = []
    for (embed, label) in query_batches:
        Q = (embed,label)
        batches.append(Q)

    return batches


def _split_into_batches(allosaurus_embedding, labels,bsize):
    batches = []
    for offset in range(0, allosaurus_embedding.shape[0], bsize):
        batches.append((allosaurus_embedding[offset:offset+bsize], labels[offset:offset+bsize]))

    return batches