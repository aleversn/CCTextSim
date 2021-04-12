import random
from torch.utils.data import TensorDataset, DataLoader, Dataset
from CC.ICCStandard import IDataLoader
from CC._dataloaders import *

class AutoDataloader(IDataLoader):

    def __init__(self, tokenizer, data_name, model_type="bert", padding_length=50):
        self.model_type = model_type
        model_list_1 = ['bert']
        if data_name == 'sim':
            if model_type in model_list_1
                self.training_set = SimDataset(tokenizer, './datasets/FNSim/train.csv', padding_length=padding_length, is_train=True)
                self.dev_set = SimDataset(tokenizer, './datasets/FNSim/dev.csv', padding_length=padding_length, mode='eval')
                self.test_set = SimDataset(tokenizer, './datasets/FNSim/test.csv', padding_length=padding_length, mode='eval')
    
    def __call__(self, batch_size=16, batch_size_eval=64, fit_sample=False):
        dataiter = DataLoader(self.training_set, batch_size=batch_size)
        dataiter_eval = DataLoader(self.dev_set, batch_size=batch_size_eval)
        dataiter_test = DataLoader(self.test_set, batch_size=batch_size_eval)
        return dataiter, dataiter_eval, dataiter_test