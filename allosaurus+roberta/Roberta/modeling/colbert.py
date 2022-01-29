import string
import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, BertTokenizerFast
from transformers import RobertaTokenizerFast, RobertaPreTrainedModel,RobertaModel

class ColBERT(RobertaPreTrainedModel):
    def __init__(self, config, query_maxlen,  mask_punctuation, num_labels=2,dim=128):

        super(ColBERT, self).__init__(config)
        self.config = config
        self.query_maxlen = query_maxlen
        self.dim = dim
        self.num_labels = num_labels

        self.mask_punctuation = mask_punctuation
        self.skiplist = {}

        if self.mask_punctuation:
            self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
            self.skiplist = {w: True
                             for symbol in string.punctuation
                             for w in [symbol, self.tokenizer.encode(symbol, add_special_tokens=False)[0]]}

        self.roberta = RobertaModel(config)


        #########          for MLM task         ##########
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = nn.ReLU()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

        #########  for SER classification task   ##########
        self.dense1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)
         

        self.init_weights()

    def MLMhead(self,hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

    def forward(self, Q, mode,label = None):
        if mode == "mlm": 
            return self.mlm_score(Q,label)
        if mode == "ser": ##classification head
            return self.classification_score(Q,label)



    def mask(self, input_ids):
        mask = [[(x not in self.skiplist) and (x != 0) for x in d] for d in input_ids.cpu().tolist()]
        return mask



    def mlm_score(self,Q,label):
        QD = Q[0].cuda() 
        QD_mask = Q[1].cuda() 
        output = self.roberta(QD, attention_mask = QD_mask)
        sequence_output = output[0]
        prediction_scores = self.MLMhead(sequence_output)
        loss_fct = torch.nn.CrossEntropyLoss()  # -100 index = padding token
        label = label.cuda()
        masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), label.view(-1))
        return masked_lm_loss

    def classification_score(self,Q,label):
        sentence = Q[0].cuda()
        sentence_mask = Q[1].cuda()
        label = label.cuda()
        output = self.roberta(sentence,attention_mask = sentence_mask)
        sequence_output = output[0][:,0,:]
        x = self.dropout(sequence_output)
        x = self.dense1(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(x.view(-1, self.num_labels), label.view(-1))
        return loss
