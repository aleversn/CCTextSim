# %%
import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, Dataset
from transformers import BertModel, BertConfig, BertTokenizer, BertForSequenceClassification

# %%
from CC.trainer import *
from transformers import BertTokenizer

# %%
tokenizer = BertTokenizer.from_pretrained('model/chinese_wwm_ext')
trainer = Trainer(tokenizer, model_dir='model/chinese_wwm_ext', dataset_name='sim', padding_length=128, batch_size=128, batch_size_eval=2000)

# %%
# Common Training
trainer.train(num_epochs=30, lr=1e-5, gpu=[0, 1, 2], eval_mode='test', is_eval=False)

# %%
trainer.save_pred(save_dir='./log', gpu=[0, 1, 2], resume_path='./model/sim/bert/epoch_8.pth')

# %%
Analysis.Evaluation('./log/pred_result.csv', './datasets/FNSim/dev.csv', 'datasets/FNSim/target_list')

# %%
Analysis.DiffOutput('./log/pred_result.csv', './datasets/FNSim/dev.csv', 'diff.csv')

# %%
from CC.predictor import *
from transformers import BertTokenizer

# %%
tokenizer = BertTokenizer.from_pretrained('model/chinese_wwm_ext')
pred = Predictor(tokenizer, model_dir='model/chinese_wwm_ext', target_file_name='./datasets/FNSim/target_list', padding_length=30, resume_path='./model/sim/bert/epoch_8.pth', batch_size=2000, gpu=[0, 1, 2, 3])

# %%
pred('频发心房下部早搏')[:5]

# %%
from tqdm import tqdm

with open('./datasets/FN_Sichuan/top100_keywords.csv', encoding='utf-8') as f:
    ori_list = f.read().split('\n')
if ori_list[len(ori_list) - 1] == '':
    ori_list = ori_list[:len(ori_list) - 1]
result = ''
for line in tqdm(ori_list):
    r = ''
    line = line.split(';')
    for entity in line:
        p = pred(entity)[0]
        r = '{};{}'.format(r, p) if r != '' else p
    result += '{}\n'.format(r)

with open('./datasets/FN_Sichuan/top100_std.csv', encoding='utf-8', mode='w+') as f:
    f.write(result)

# %%
