import torch
import torch.nn as nn

class SER_MODEL(nn.Module):
    def __init__(self, audio_maxlen, num_labels=4, hidden_size = 120,dim=128):

        super(SER_MODEL, self).__init__()
        self.audio_maxlen = audio_maxlen
        self.dim = dim
        self.num_labels = num_labels

        #########  for SER classification task   ##########
        self.dense1 = nn.Linear(230, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(hidden_size, num_labels)
         

    def forward(self, feat_emb, label = None):
        return self.classification_score(feat_emb, label)

    def classification_score(self,feat_emb,label):
        x = torch.sum(feat_emb,axis = 1)
        x = self.dense1(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        loss_fct = torch.nn.CrossEntropyLoss()
        label = label.long()
        loss = loss_fct(x.view(-1, self.num_labels), label.view(-1))
        return loss
