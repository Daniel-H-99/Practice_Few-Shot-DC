import torch
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

import FSLTask

def distribution_calibration(query, base_means, base_cov, k, alpha=0.21):
    # TODO : fill this method 
    return calibrated_mean, calibrated_cov

if __name__ == '__main__':
    # ---- data loading
    dataset = 'CUB'
    n_shot = 1
    n_ways = 5
    n_queries = 15
    n_runs = 100
    n_lsamples = n_ways * n_shot
    n_usamples = n_ways * n_queries
    n_samples = n_lsamples + n_usamples

    # Load Few-Shot Learning tasks
    cfg = {'shot': n_shot, 'ways': n_ways, 'queries': n_queries}
    FSLTask.loadDataSet(dataset)
    ndatas = FSLTask.GenerateRunSet(end=n_runs, cfg=cfg)
    ndatas = ndatas.permute(0, 2, 1, 3).reshape(n_runs, n_samples, -1)
    labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries, 5).clone().view(n_runs,
                                                                                                        n_samples)
    # ---- Base class statistics
    base_means, base_cov = [], []
    base_features_path = "./checkpoints/%s/WideResNet28_10_S2M2_R/last/base_features.plk" % dataset

    with open(base_features_path, 'rb') as f:
        data = pickle.load(f)
        for key in data.keys():
            feature = np.array(data[key])
            # TODO : calculate mean and covariance

    # ---- classification for each task
    acc_list = []
    for i in tqdm(range(n_runs)):
        # TODO
        acc_list.append(acc)
    print('%s %d way %d shot  ACC : %f'%(dataset, n_ways, n_shot, float(np.mean(acc_list))))