import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from SER.modeling.GE2E_model import SpeakerEncoder

#state_fpath = "/data1/jiayu_xiao/project/wzh_recommendation/Speech-Emotion-Recognition/encoder.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SER_MODEL(nn.Module):
    def __init__(self, args,audio_maxlen, num_labels, hidden_size = 128):
        if args.langs == "pe":
            ALLO_CONV_SIZE = 64
            ALLO_LSTM_SIZE = 64
            ALLO_ATTN_SIZE = 64
            ALLO_LSTM_NUM = 2

            MFCC_CONV_SIZE = 32
            MFCC_LSTM_SIZE = 64
            MFCC_LSTM_NUM = 1

            DROP_OUT = 0.05
            hidden_size = 400
        if args.langs == "ge" or args.langs == "en":
            ALLO_CONV_SIZE = 128
            ALLO_LSTM_SIZE = 128
            ALLO_ATTN_SIZE = 128
            ALLO_LSTM_NUM = 1

            MFCC_CONV_SIZE = 32
            MFCC_LSTM_SIZE = 64
            MFCC_LSTM_NUM = 1

            DROP_OUT = 0.1
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
        '''
        self.SpeakerEncoder = SpeakerEncoder("cpu","cpu")  ####small learning rate
        checkpoint = torch.load(state_fpath, map_location='cpu')
        self.SpeakerEncoder.load_state_dict(checkpoint["model_state"])
        print(self.SpeakerEncoder.cuda())
        '''

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

    def forward(self, feat_emb, ge2e_emb,mfcc_emb,label,length,mfcc_length):
        
        return self.classification_score(feat_emb,ge2e_emb,mfcc_emb,label,length,mfcc_length)


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


    def classification_score(self,feat_emb,ge2e_emb,mfcc_emb,label,length,mfcc_length):
        
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
            ge2e_emb = ge2e_emb.squeeze()
            x = torch.concat((mfcc_allo_hidden_state, ge2e_emb),1)
        else:
            x = mfcc_allo_hidden_state
        
        if self.args.only_GE2E:
            x = ge2e_emb.squeeze()
        if self.args.only_MFCC:
            x = mfcc_hidden_state
        ############## MLP ################
        x = self.dense1(x)
        x = self.bn(x)
        m = nn.Mish()
        #x = F.gelu(x)
        x = torch.tanh(x)
        if self.args.langs == "pe":
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
        return loss,x
