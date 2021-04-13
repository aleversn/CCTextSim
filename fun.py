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
trainer = Trainer(tokenizer, model_dir='model/chinese_wwm_ext', dataset_name='sim', padding_length=128, batch_size=128)

# %%
# Common Training
trainer.train(num_epochs=30, lr=5e-5, gpu=[0, 1, 2, 3], eval_mode='test', is_eval=False)

# %%
