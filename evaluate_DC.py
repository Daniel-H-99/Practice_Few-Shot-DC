import os
import time
import logging
from tqdm import tqdm

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import *
from sklearn.linear_model import LogisticRegression

import configs
import FSLTask
from models import wrn28_10
from io_utils import parse_args, get_best_file
from save_plk import WrappedModel



def distribution_calibration(query, base_means, base_cov, k, alpha=0.21):
    # TODO : fill this method 
    # query: (n_lsamples x feature_dim), base_means: (n_bsamples x feature_dim)
    # distances: (n_lsamples x n_bsamples)
    spent_time = time.time()
    neighborhoods = (query.unsqueeze(1) - base_means.unsqueeze(0)).pow(2).sum(dim=2).argsort(dim=1)[:, :k]
    calibrated_mean = torch.cat((base_means[neighborhoods], query.unsqueeze(1)), dim=1).mean(dim=1)
    calibrated_cov = base_cov[neighborhoods].mean(dim=1) + alpha
    return calibrated_mean, calibrated_cov

if __name__ == '__main__':
    args = parse_args('eval')
    
    if not os.path.isdir(os.path.join(args.path, './ckpt')):
        os.mkdir(os.path.join(args.path,'./ckpt'))
    if not os.path.isdir(os.path.join(args.path,'./results')):
        os.mkdir(os.path.join(args.path,'./results'))    
    if not os.path.isdir(os.path.join(args.path, './ckpt', args.name)):
        os.mkdir(os.path.join(args.path, './ckpt', args.name))
    if not os.path.isdir(os.path.join(args.path, './results', args.name)):
        os.mkdir(os.path.join(args.path, './results', args.name))
    if not os.path.isdir(os.path.join(args.path, './results', args.name, "log")):
        os.mkdir(os.path.join(args.path, './results', args.name, "log"))

    # set logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler = logging.FileHandler(os.path.join(args.path, "results/{}/log/{}.log".format(args.name, time.strftime('%c', time.localtime(time.time())))))
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())
    args.logger = logger
    
    args.logger.info("[{}] starts".format(args.name))
    
    # ---- data loading
    dataset = 'CUB'
    n_shot = 1
    n_ways = 5
    n_queries = 15
    n_runs = 100
    n_lsamples = n_ways * n_shot
    n_usamples = n_ways * n_queries
    n_samples = n_lsamples + n_usamples
    k = args.k
    feature_dim = 640
    n_generated_features = args.sampling
    alpha = 0.3
    lamda = args.lamda
    checkpoint_dir = "./ckpt/%s/WideResNet28_10_S2M2_R" % dataset
    
    # Load Few-Shot Learning tasks
    cfg = {'shot': n_shot, 'ways': n_ways, 'queries': n_queries}
    FSLTask.loadDataSet(dataset)
    ndatas = FSLTask.GenerateRunSet(end=n_runs, cfg=cfg)
    ndatas = ndatas.permute(0, 2, 1, 3).reshape(n_runs, n_samples, -1)
    labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries, 5).clone().view(n_runs,
                                                                                                        n_samples)
    # ---- Base class statistics
    base_means, base_cov = [], []
    base_features_path = checkpoint_dir + "/last/base_features.plk"

    with open(base_features_path, 'rb') as f:
        data = pickle.load(f)
        for key in data.keys():
            features = np.array(data[key])
            # TODO : calculate mean and covariance
            base_means.append(features.mean(0))
            base_cov.append(np.cov(features.T))
            
    # ---- classification for each task
    acc_list = []
    
    for i in tqdm(range(n_runs)):
        # TODO
        print("constructing dataset...")
        datas = torch.Tensor(ndatas[i]).pow(lamda)
        classes = torch.Tensor(FSLTask.ClassesInRun(i, cfg))[labels[i]]   # (n_samples)
        supports, queries = datas[:n_lsamples], datas[n_lsamples:n_samples]
        classes_supports, classes_queries = classes[:n_lsamples], classes[n_lsamples:n_samples]
        
        print("calibrating distributions...")
        spent_time = time.time()
        cali_means, cali_cov = distribution_calibration(supports, torch.Tensor(base_means), torch.Tensor(base_cov), k, alpha=alpha)
        concated_cali_means = cali_means.view(-1)
        concated_cali_cov = torch.zeros(n_lsamples, n_lsamples, feature_dim, feature_dim)
        concated_cali_cov[torch.eye(n_lsamples, n_lsamples).nonzero(as_tuple=True)] = cali_cov
        concated_cali_cov = torch.cat(torch.cat(concated_cali_cov.split(1), dim=2).split(1, dim=1), dim=3).squeeze(0).squeeze(0)
        args.logger.info("D.C took {:.3f} sesconds".format(time.time() - spent_time))
        
        print("generating features...")
        samples_per_stats = n_generated_features // n_shot
        generated_features = torch.Tensor(np.random.multivariate_normal(mean=concated_cali_means, cov=concated_cali_cov, size=samples_per_stats)).view(-1, feature_dim)
        generated_classes = classes_supports.repeat(samples_per_stats)
        
        print("preparing training...")
        train_features = np.concatenate([supports.numpy(), generated_features.numpy()])
        train_classes = np.concatenate([classes_supports.numpy(), generated_classes.numpy()])
        model = LogisticRegression(max_iter=1000).fit(X=train_features, y=train_classes)
        acc = (model.predict(queries.numpy()) == classes_queries.numpy()).mean()
        acc_list.append(acc)
        args.logger.info('[{} accuracy: {}'.format(i + 1, acc))
    args.logger.info('%s %d way %d shot  ACC : %f'%(dataset, n_ways, n_shot, float(np.mean(acc_list))))