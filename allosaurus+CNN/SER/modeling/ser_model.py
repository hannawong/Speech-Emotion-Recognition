import torch
import torch.nn as nn
import torch.nn.functional as F

class SER_MODEL(nn.Module):
    def __init__(self, audio_maxlen, num_labels=4, hidden_size = 64):

        super(SER_MODEL, self).__init__()
        self.audio_maxlen = audio_maxlen
        self.num_labels = num_labels

        #########  for SER classification task   ##########
        self.conv1d_1 = nn.Conv1d(in_channels=230, out_channels=128, kernel_size=3,padding = "same")
        self.conv1d_2 = nn.Conv1d(in_channels=230, out_channels=128, kernel_size=5,padding = "same")
        self.lstm = nn.LSTM(128, 128, 1, bidirectional=True)
        self.dense1 = nn.Linear(128*2, hidden_size*4)
        self.bn = torch.nn.BatchNorm1d(hidden_size*4)
        self.dropout = nn.Dropout(0.1)
        self.dense2 = nn.Linear(hidden_size*4,hidden_size)
        self.dense3 = nn.Linear(hidden_size,hidden_size // 2)
        self.out_proj = nn.Linear(hidden_size//2, num_labels)

         

    def forward(self, feat_emb, ge2e_emb,label = None):
        return self.classification_score(feat_emb,ge2e_emb,label)

    def classification_score(self,feat_emb,ge2e_emb,label):
        feat_emb = feat_emb.permute(0, 2, 1)
        print(feat_emb.shape)
        x = self.conv1d_1(feat_emb)
        x1 = self.conv1d_2(feat_emb)
        x = x.permute(0,2,1)
        x1 = x1.permute(0,2,1)
        x = torch.add(x,x1)
        x,_ = self.lstm(x)
        x = torch.mean(x,axis = 1)
        
        ge2e_emb = ge2e_emb.squeeze()
        x = torch.add(x,ge2e_emb)

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