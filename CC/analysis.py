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
        if len(X) == 0:
            return 0, 0
        if len(X) != len(Y):
            raise Exception('mismatch length of X and Y')
        x_mean = np.mean(X)
        y_mean = np.mean(Y)
        r_a = 0
        r_b = 0
        r_c = 0
        r_mse = 0
        for i in range(len(X)):
            _x = (X[i] - x_mean)
            _y = (Y[i] - y_mean)
            r_a += _x * _y
            r_b += _x ** 2
            r_c += _y ** 2
            r_mse += (X[i] - Y[i]) ** 2
        r = r_a / (r_b ** 0.5 * r_c ** 0.5)
        r_mse = (r_mse / len(X)) ** 0.5

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
