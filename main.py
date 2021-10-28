import numpy as np
from numpy.random import default_rng
from Decision_tree import DecisionTree
from utils import n_fold_split, cross_validation, nested_cross_validation
from metric import MyMetric


def load_data(data_path):
    return np.loadtxt(data_path)

def train(training_data, n=10, seed=1024, prune=False):
    n_folds = n_fold_split(training_data, n, default_rng(seed=seed))
    if prune:
        tr, metric, depth_list = nested_cross_validation(DecisionTree, n_folds, measure='f1')
    else:
        tr, metric, depth_list = cross_validation(DecisionTree, n_folds)
    return tr, metric, depth_list

def mean_depth(depth_list):
    result = 0.0
    for depth in depth_list:
        result += depth
    result /= len(depth_list)
    return result

if __name__ == '__main__':
    print('-'*30)
    print('Loading Data...')
    data_clean = load_data('wifi_db/clean_dataset.txt')
    data_noisy = load_data('wifi_db/noisy_dataset.txt')


    print('-'*30)
    print('Training cleaning data for part 3:')
    tree, metric, depth_list = train(data_clean, 10, 1024, False)
    print('Result of part 2 with clean data:')  
    print(metric.get_metric())
    print('Depth for each tree:')
    print(depth_list)
    print('Avarage of depth:')
    print(str(mean_depth(depth_list)))

    print('-'*30)
    print('Training noisy data for part 3:')
    tree, metric, depth_list = train(data_noisy, 10, 1024, False)
    print('Result of part 2 with noisy data:')
    print(metric.get_metric())
    print('Depth for each tree:')
    print(depth_list)
    print('Avarage of depth:')
    print(str(mean_depth(depth_list)))

    print('-'*30)
    print('Training clean data for part 4:')
    tree, metric, depth_list = train(data_clean, 10, 1024, True)
    print('Result of part 3 with clean data:')
    print(metric.get_metric())
    print('Depth for each tree:')
    print(depth_list)
    print('Avarage of depth:')
    print(str(mean_depth(depth_list)))

    print('-'*30)
    print('Training noisy data for part 4:')
    tree, metric, depth_list = train(data_noisy, 10, 1024, True)
    print('Result of part 3 with noisy data:')
    print(metric.get_metric())
    print('Depth for each tree:')
    print(depth_list)
    print('Avarage of depth:')
    print(str(mean_depth(depth_list)))

