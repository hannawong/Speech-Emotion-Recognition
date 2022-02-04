from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.nn.functional as F

class SER_MODEL(nn.Module):
    def __init__(self, audio_maxlen, num_labels=4, hidden_size = 128):

        super(SER_MODEL, self).__init__()
        self.audio_maxlen = audio_maxlen
        self.num_labels = num_labels

        #########  for SER classification task   ##########
        self.lstm = nn.LSTM(230, 230, 1, bidirectional=False)
        self.dense1 = nn.Linear(230, hidden_size*4)
        self.bn = torch.nn.BatchNorm1d(hidden_size*4)
        self.dropout = nn.Dropout(0.1)
        self.dense2 = nn.Linear(hidden_size*4,hidden_size)
        self.dense3 = nn.Linear(hidden_size,hidden_size // 2)
        self.out_proj = nn.Linear(hidden_size//2, num_labels)

         

    def forward(self, feat_emb, label = None):
        return self.classification_score(feat_emb, label)

    def classification_score(self,feat_emb,label):
        x,_ = self.lstm(feat_emb)
        x = torch.sum(x,axis = 1)
        x = F.normalize(x,p=2,dim=1) 
        x = self.dense1(x)
        x = self.bn(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = F.relu(x)
        x = self.dense3(x)
        x = F.relu(x)
        x = self.out_proj(x)
        loss_fct = torch.nn.CrossEntropyLoss()
        label = label.long()
        loss = loss_fct(x.view(-1, self.num_labels), label.view(-1))
        return loss,x
