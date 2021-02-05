from __future__ import print_function

import os
import pickle
import collections

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import configs
from models import wrn28_10
from data.datamgr import SimpleDataManager, SetDataManager
from io_utils import parse_args, get_resume_file

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- pickle related methods
def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module 

    def forward(self, x):
        return self.module(x)

# ---- main code 
def extract_feature(val_loader, model, checkpoint_dir, tag='last',set='base'):
    save_dir = '{}/{}'.format(checkpoint_dir, tag)
    if os.path.isfile(save_dir + '/%s_features.plk'%set):
        data = load_pickle(save_dir + '/%s_features.plk'%set)
        return data
    else:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    model.eval()
    with torch.no_grad():
        
        output_dict = collections.defaultdict(list)

        for i, (inputs, labels) in enumerate(val_loader):
            # compute output
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)
            outputs = outputs.cpu().data.numpy()
            
            for out, label in zip(outputs, labels):
                output_dict[label.item()].append(out)
    
        all_info = output_dict
        save_pickle(save_dir + '/%s_features.plk'%set, all_info)
        return all_info

if __name__ == '__main__':
    params = parse_args('test')
    loadfile_base = configs.data_dir[params.dataset] + 'base.json'
    loadfile_novel = configs.data_dir[params.dataset] + 'novel.json'
    
    datamgr = SimpleDataManager(84, batch_size = 256)
    base_loader = datamgr.get_data_loader(loadfile_base, aug= False)
    novel_loader = datamgr.get_data_loader(loadfile_novel, aug= False)

    checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
    modelfile = get_resume_file(checkpoint_dir)

    model = wrn28_10(num_classes=params.num_classes)
    model = model.to(device)

    cudnn.benchmark = True

    checkpoint = torch.load(modelfile)
    state = checkpoint['state']
    state_keys = list(state.keys())

    callwrap = False
    if 'module' in state_keys[0]:
        callwrap = True
    if callwrap:
        model = WrappedModel(model)

    model_dict_load = model.state_dict()
    model_dict_load.update(state)
    model.load_state_dict(model_dict_load)

    output_dict_base = extract_feature(base_loader, model, checkpoint_dir, tag='last', set='base')
    print("base set features saved!")
    output_dict_novel=extract_feature(novel_loader, model, checkpoint_dir, tag='last',set='novel')
    print("novel features saved!")
