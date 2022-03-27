import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from SER.modeling.GE2E_model import SpeakerEncoder
from SER.modeling.audio import *
from SER.modeling.inference import *
import pickle as pkl
import pandas as pd
from tqdm import tqdm
dic = pkl.load(open("/content/drive/MyDrive/warm_up_pkls/ge2e_input.pkl","rb"))
print("warm up finish!")
state_fpath = "/content/drive/MyDrive/Speech-Emotion-Recognition/allosaurus+CNN/SER/modeling/encoder.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
'''
#=========================
data = pd.read_csv("/content/drive/MyDrive/Speech-Emotion-Recognition/allosaurus+CNN/persian/total.csv")
data1 = pd.read_csv("/content/drive/MyDrive/Speech-Emotion-Recognition/allosaurus+CNN/emodb/total.csv")

data2 = pd.read_csv("/content/drive/MyDrive/Speech-Emotion-Recognition/allosaurus+CNN/iemocap/_iemocap_01F.test.csv")
data3 = pd.read_csv("/content/drive/MyDrive/Speech-Emotion-Recognition/allosaurus+CNN/iemocap/_iemocap_01F.train.csv")
data4 = pd.read_csv("/content/drive/MyDrive/Speech-Emotion-Recognition/allosaurus+CNN/iemocap/_iemocap_01F.val.csv")

for file in tqdm(list(data3[data3.columns[0]])):
  try:
    wav = preprocess_wav("/content/drive/MyDrive/path_to_wavs/"+file.split("/")[-1])
    dic[file] = wav
  except:
    print(file)


for file in tqdm(list(data2[data2.columns[0]])):
  try:
    wav = preprocess_wav("/content/drive/MyDrive/path_to_wavs/"+file.split("/")[-1])
    dic[file] = wav
  except:
    print(file)

for file in tqdm(list(data4[data4.columns[0]])):
  try:
    wav = preprocess_wav("/content/drive/MyDrive/path_to_wavs/"+file.split("/")[-1])
    dic[file] = wav
  except:
    print(file)
for file in tqdm(list(data[data.columns[1]])+list(data1[data1.columns[1]])):
  try:
    wav = preprocess_wav("/content/drive/MyDrive/path_to_wavs/"+file.split("/")[-1])
    dic[file] = wav
  except:
    print(file)

output = open('/content/drive/MyDrive/warm_up_pkls/ge2e_input.pkl', 'wb')
pkl.dump(dic, output)
'''
#====================

              
              


class SER_MODEL(nn.Module):
    def __init__(self, args,audio_maxlen, num_labels, hidden_size = 128):

        if args.langs == "pe":
            ALLO_CONV_SIZE = 32
            ALLO_LSTM_SIZE = 32
            ALLO_ATTN_SIZE = 32
            ALLO_LSTM_NUM = 2

            MFCC_CONV_SIZE = 32
            MFCC_LSTM_SIZE = 64
            MFCC_LSTM_NUM = 1

            DROP_OUT = 0.01
            hidden_size = 512
        if args.langs == "ge": 
            ALLO_CONV_SIZE = 128
            ALLO_LSTM_SIZE = 128
            ALLO_ATTN_SIZE = 128
            ALLO_LSTM_NUM = 1

            MFCC_CONV_SIZE = 32
            MFCC_LSTM_SIZE = 64
            MFCC_LSTM_NUM = 1

            DROP_OUT = 0.05
            hidden_size = 256
        if args.langs == "en":
            ALLO_CONV_SIZE = 128
            ALLO_LSTM_SIZE = 128
            ALLO_ATTN_SIZE = 128
            ALLO_LSTM_NUM = 1

            MFCC_CONV_SIZE = 32
            MFCC_LSTM_SIZE = 64
            MFCC_LSTM_NUM = 1

            DROP_OUT = 0.05
            hidden_size = 256


        super(SER_MODEL, self).__init__()
        self.audio_maxlen = audio_maxlen
        self.num_labels = num_labels
        self.args = args
        #########  Allosaurus feature   ##########
        self.conv1d_1 = nn.Conv1d(in_channels=230, out_channels=ALLO_CONV_SIZE, kernel_size=3,padding = "same")
        self.conv1d_2 = nn.Conv1d(in_channels=230, out_channels=ALLO_CONV_SIZE, kernel_size=5,padding = "same")
        self.lstm = nn.LSTM(ALLO_CONV_SIZE, ALLO_LSTM_SIZE, ALLO_LSTM_NUM, bidirectional=True)
        self.Q_layer = nn.Linear(ALLO_LSTM_SIZE*2,ALLO_ATTN_SIZE)
        self.K_layer = nn.Linear(ALLO_LSTM_SIZE*2,ALLO_ATTN_SIZE)
        self.V_layer = nn.Linear(ALLO_LSTM_SIZE*2,ALLO_ATTN_SIZE)

        ############# for MFCC ###############
        self.conv1d_1_mfcc = nn.Conv1d(in_channels=24, out_channels=MFCC_CONV_SIZE, kernel_size=3,padding = "same")
        self.conv1d_2_mfcc = nn.Conv1d(in_channels=24, out_channels=MFCC_CONV_SIZE, kernel_size=5,padding = "same")
        self.lstm_mfcc = nn.LSTM(MFCC_CONV_SIZE,MFCC_LSTM_SIZE, MFCC_LSTM_NUM, bidirectional=True)
        self.Q_layer_mfcc = nn.Linear(MFCC_LSTM_SIZE*2,ALLO_ATTN_SIZE)
        self.K_layer_mfcc = nn.Linear(MFCC_LSTM_SIZE*2,ALLO_ATTN_SIZE)
        self.V_layer_mfcc = nn.Linear(MFCC_LSTM_SIZE*2,ALLO_ATTN_SIZE)

        ############# for GE2E finetuning #############
        
        self.SpeakerEncoder = SpeakerEncoder('cpu','cpu')  ####small learning rate
        checkpoint = torch.load(state_fpath, map_location='cpu')
        self.SpeakerEncoder.load_state_dict(checkpoint["model_state"])
        self.SpeakerEncoder = self.SpeakerEncoder.to(DEVICE)
      

        ########## MLP ############
        if args.GE2E:
            self.dense1 = nn.Linear(256+ALLO_ATTN_SIZE, hidden_size*2)
        else:
            self.dense1 = nn.Linear(ALLO_ATTN_SIZE, hidden_size*2)
        if args.only_GE2E:
            self.dense1 = nn.Linear(256,hidden_size*2)
        self.bn = torch.nn.BatchNorm1d(hidden_size*2)
        self.bn1 = torch.nn.BatchNorm1d(hidden_size)
        self.bn2 = torch.nn.BatchNorm1d(hidden_size//2)
        self.dropout = nn.Dropout(DROP_OUT)
        self.dense2 = nn.Linear(hidden_size*2,hidden_size)
        self.dense3 = nn.Linear(hidden_size,hidden_size // 2)
        self.out_proj = nn.Linear(hidden_size//2, num_labels)

    def forward(self, feat_emb, ge2e_emb,mfcc_emb,label,length,mfcc_length,audio_files):
        
        return self.classification_score(feat_emb,ge2e_emb,mfcc_emb,label,length,mfcc_length,audio_files)


    def get_attention_mask(self,lstm_output,length):
        length = np.array(length.detach().cpu())
        MAX_LEN = lstm_output.shape[1]
        BZ = lstm_output.shape[0]
        mask = [[1]*int(length[_])+[0]*(MAX_LEN-int(length[_])) for _ in range(BZ)]
        mask = torch.Tensor(mask)
        return mask
    

    def self_attention_layer(self,lstm_output,length):  # TODO: multi-head self-attention
        last_hidden_state = [lstm_output[i,length[i].long(),:] for i in range(lstm_output.shape[0])] ## the actual hidden state!
        last_hidden_state = torch.stack(last_hidden_state,axis = 0)
        Q_last_hidden_state = self.Q_layer(last_hidden_state) ##Query, (batchsize,128)
        Q_last_hidden_state = torch.unsqueeze(Q_last_hidden_state,1)
        K = self.K_layer(lstm_output) ##(batchsize,max_len,128)
        V = self.V_layer(lstm_output) ##(batchsize,max_len,128)
        attention_scores = torch.matmul(Q_last_hidden_state,K.permute(0,2,1))
        attention_scores = torch.multiply(attention_scores,
                                   1.0 / math.sqrt(float(attention_scores.shape[-1])))

        attention_mask = self.get_attention_mask(lstm_output,length) ##can only attend to hidden state that really exist
        adder = (1.0 - attention_mask.long()) * -10000.0  ##-infty, [batchsize,150]
        adder = torch.unsqueeze(adder,axis = 1).to(DEVICE)
        attention_scores += adder

        m = nn.Softmax(dim=2)
        attention_probs = m(attention_scores)
        context_layer = torch.matmul(attention_probs, V)
        return context_layer.squeeze()

    def self_attention_layer_mfcc(self,lstm_output,length):  # TODO: multi-head self-attention
        last_hidden_state = [lstm_output[i,length[i].long(),:] for i in range(lstm_output.shape[0])] ## the actual hidden state!
        last_hidden_state = torch.stack(last_hidden_state,axis = 0)
        Q_last_hidden_state = self.Q_layer_mfcc(last_hidden_state) ##Query, (batchsize,128)
        Q_last_hidden_state = torch.unsqueeze(Q_last_hidden_state,1)
        K = self.K_layer_mfcc(lstm_output) ##(batchsize,max_len,128)
        V = self.V_layer_mfcc(lstm_output) ##(batchsize,max_len,128)
        attention_scores = torch.matmul(Q_last_hidden_state,K.permute(0,2,1))
        attention_scores = torch.multiply(attention_scores,
                                   1.0 / math.sqrt(float(attention_scores.shape[-1])))

        attention_mask = self.get_attention_mask(lstm_output,length) ##can only attend to hidden state that really exist
        adder = (1.0 - attention_mask.long()) * -10000.0  ##-infty, [batchsize,150]
        adder = torch.unsqueeze(adder,axis = 1).to(DEVICE)
        attention_scores += adder

        m = nn.Softmax(dim=2)
        attention_probs = m(attention_scores)
        context_layer = torch.matmul(attention_probs, V)
        return context_layer.squeeze()

    def construct_matrix_embed(self,embeds,labels):
      label_cnt = {}
      for i in range(len(labels)):
        if int(labels[i].item()) not in label_cnt:
          label_cnt[int(labels[i].item())] = 1
        else:
          label_cnt[int(labels[i].item())] += 1
      label_cnt_pruned = {} ###only retain those cnt >= 5
      for k in label_cnt.keys():
        if label_cnt[k] >= self.args.bsize // self.num_labels - 3:
          label_cnt_pruned[k] = label_cnt[k]
      speaker_per_batch = len(label_cnt_pruned)
      utterances_per_speaker = np.min(list(label_cnt_pruned.values()))
      embeddings = np.zeros((speaker_per_batch,utterances_per_speaker,embeds.shape[-1]))
      for k in range(len(label_cnt_pruned.keys())): ### emotions
        key = list(label_cnt_pruned.keys())[k]
        cnt = 0
        for i in range(len(labels)):
          if int(labels[i].item()) == key and cnt < utterances_per_speaker:
            embeddings[k][cnt] = embeds[i].cpu().detach().numpy()
            cnt += 1
      return torch.Tensor(embeddings)

    def classification_score(self,feat_emb,ge2e_emb,mfcc_emb,label,length,mfcc_length,audio_files):
        
        ########## allosaurus features #############
        feat_emb = feat_emb.permute(0, 2, 1)
        x = self.conv1d_1(feat_emb)
        x1 = self.conv1d_2(feat_emb)
        x = x.permute(0,2,1)
        x1 = x1.permute(0,2,1)
        x = torch.add(x,x1)
        x,_ = self.lstm(x)
        allo_hidden_state = self.self_attention_layer(x,length) ##[bz,128]

        ################## mfcc features ####################
        mfcc = mfcc_emb.permute(0, 2, 1)
        mfcc_x = self.conv1d_1_mfcc(mfcc)
        mfcc_x1 = self.conv1d_2_mfcc(mfcc)
        mfcc_x = mfcc_x.permute(0,2,1)
        mfcc_x1 = mfcc_x1.permute(0,2,1)
        mfcc_x = torch.add(mfcc_x,mfcc_x1)
        mfcc_x,_ = self.lstm_mfcc(mfcc_x)
        mfcc_hidden_state = self.self_attention_layer_mfcc(mfcc_x,mfcc_length) ##[bz,32]
        
        ############### combine mfcc and allosaurus ###########
        if self.args.MFCC and not self.args.no_Allo:
            mfcc_allo_hidden_state = torch.add(allo_hidden_state,mfcc_hidden_state)
        elif not self.args.MFCC and not self.args.no_Allo:
            mfcc_allo_hidden_state = allo_hidden_state
        else:
            mfcc_allo_hidden_state = mfcc_hidden_state
        ############### add ge2e #####################
        if self.args.GE2E:
            ge2e_emb = ge2e_emb.squeeze()  ### hard-encode ge2e_emb
            
            ge2e_emb = []
            for i in range(len(audio_files)):
              if audio_files[i] not in dic:
                wav = preprocess_wav("/content/drive/MyDrive/path_to_wavs/"+audio_files[i].split("/")[-1])
                dic[audio_files[i]] = wav
              else:
                wav = dic[audio_files[i]]
              emb = embed_utterance(wav,self.SpeakerEncoder.to(DEVICE))
              ge2e_emb.append(torch.Tensor(emb))
            ge2e_emb = torch.stack(ge2e_emb, axis = 0)
            ge2e_emb = ge2e_emb.to(DEVICE)
            print(ge2e_emb.shape)
          
            
            
            x = torch.concat((mfcc_allo_hidden_state, ge2e_emb),1)
        else:
            x = mfcc_allo_hidden_state
        
        if self.args.only_GE2E:
            x = ge2e_emb.squeeze()
        if self.args.only_MFCC:
            x = mfcc_hidden_state
        ############## MLP ################
        matrix_embedding = self.construct_matrix_embed(x,label).to(DEVICE)
        contrastive_loss,err = self.SpeakerEncoder.loss(matrix_embedding)
        print("contrastive loss:",contrastive_loss,err)

        x = self.dense1(x)
        x = self.bn(x)
        m = nn.Mish()
        #x = F.gelu(x)
        
        x = torch.tanh(x)
        if self.args.langs in ["pe"]:
          x = m(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.bn1(x)
        
        x = F.relu(x)
        if self.args.langs == "pe":
          x = F.gelu(x)
        x = self.dropout(x)
        x = self.dense3(x)
        x = self.bn2(x)
        x = F.gelu(x)
        x = self.out_proj(x)
        loss_fct = torch.nn.CrossEntropyLoss()
        label = label.long()
        loss = loss_fct(x.view(-1, self.num_labels), label.view(-1))
        alpha = 0.3
        total_loss = loss+alpha*contrastive_loss
        return total_loss,x
