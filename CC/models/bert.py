import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertTokenizer, BertForSequenceClassification

class Bert(nn.Module):

    def __init__(self, tokenizer, model_pretrained_dir):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained(model_pretrained_dir)
        self.tokenizer = tokenizer
    
    def forward(self, sentences, attention_mask, token_type_ids, labels, padding_length):
        fct_loss = nn.MSELoss()
        outputs = self.model(sentences, attention_mask=attention_mask, token_type_ids=token_type_ids)

        logits = outputs[0]
        p = F.softmax(logits, dim=-1)
        pred = p[:, 1]

        loss = fct_loss(p[:, 1], labels.view(-1))

        return loss, pred