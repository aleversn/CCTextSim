import os
import datetime
import numpy as np
from scipy.stats import pearsonr, spearmanr
from CC.ICCStandard import IAnalysis

class Analysis(IAnalysis):

    def __init__(self):
        print(0)
    
    @staticmethod
    def WriteSDC(name, info):
        with open("./log/{}.txt".format(name), mode="a+", encoding="utf-8") as f:
            f.write(info)
    
    @staticmethod
    def Evaluation(X, Y):
        correct = 0
        pos = 0
        total_P = len(X)
        for idx, x in enumerate(X):
            if x == Y[idx]:
                correct += 1

        return r, r_mse, pearsonr(X, Y)[0], spearmanr(X, Y)[0]
    
    @staticmethod
    def heatmap(data):
        return ValueError('')
    
    @staticmethod
    def save_xy(X, Y, dir):
        if not os.path.isdir(dir):
            os.makedirs(dir)
        result = ''
        for i in range(len(X)):
            result += '{}\t{}\n'.format(X[i], Y[i])
        with open('{}/predict_gold.csv'.format(dir), encoding='utf-8', mode='w+') as f:
            f.write(result)
    
    @staticmethod
    def save_same_row_list(dir, file_name, **args):
        if not os.path.isdir(dir):
            os.makedirs(dir)
        result = ''
        dicts = []
        for key in args.keys():
            dicts.append(key)
            result = key if result == '' else result + '\t{}'.format(key)
        length = len(args[dicts[0]])
        result += '\n'
        for i in range(length):
            t = True
            for key in args.keys():
                result += str(args[key][i]) if t else '\t{}'.format(args[key][i])
                t = False
            result += '\n'
        with open('{}/{}.csv'.format(dir, file_name), encoding='utf-8', mode='w+') as f:
            f.write(result)
