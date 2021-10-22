import numpy as np
from numpy.random import default_rng
from Decision_tree import DecisionTree
import utils
from utils import n_fold_split, cross_validation, visualize
from metric import MyMetric


def load_data(data_path):
    return np.loadtxt(data_path)

def train(training_data, n=10, seed=1024, prune=False):
    n_folds = n_fold_split(training_data, n, default_rng(seed=seed))
    tr, metric = cross_validation(DecisionTree, n_folds, prune=prune)
    #visualize(tr.root,"report.PNG",BOX_SIZE=6)
    return tr, metric



if __name__ == '__main__':
    print('-'*30)
    print('Loading Data...')
    data = load_data('wifi_db/clean_dataset.txt')
    print('-'*30)
    print('Training')
    tree, metric = train(data, 10, 1024, True)
    print('-'*30)  
    print(metric.get_metric())
