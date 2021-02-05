#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
from __future__ import print_function

import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable

from models import wrn28_10
from io_utils import parse_args
from configs import save_dir, data_dir
from data.datamgr import SimpleDataManager

image_size = 84
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_s2m2(base_loader_test, model, criterion):
    model.eval()
    with torch.no_grad():
        test_loss = 0
        correct, total = 0, 0
        for batch_idx, (inputs, targets) in enumerate(base_loader_test):
            inputs, targets = Variable(inputs.to(device)), Variable(targets.to(device))
            f, outputs = model.forward(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.data.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            print('Loss: %.3f | Acc: %.3f%%' % (test_loss/(batch_idx+1), 100.*correct/total)) 

def train_s2m2(base_loader, base_loader_test, model, params, tmp):

    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    criterion = nn.CrossEntropyLoss()

    rotate_classifier = nn.Sequential(nn.Linear(640,4))
    rotate_classifier.to(device)
    model.to(device)

    if 'rotate' in tmp:
        print("loading rotate model")
        rotate_classifier.load_state_dict(tmp['rotate'])

    optimizer = torch.optim.Adam([
                {'params': model.parameters()},
                {'params': rotate_classifier.parameters()}
            ])
    
    start_epoch, stop_epoch = params.start_epoch, params.start_epoch+params.stop_epoch
    print("stop_epoch", start_epoch, stop_epoch)

    for epoch in range(start_epoch, stop_epoch):
        print('\nEpoch: %d' % epoch)

        model.train()
        train_loss, rotate_loss = 0, 0
        correct, total = 0, 0
        torch.cuda.empty_cache()
        
        for batch_idx, (inputs, targets) in enumerate(base_loader):
            
            inputs, targets = inputs.to(device), targets.to(device)
            lam = np.random.beta(params.alpha, params.alpha)
            f, outputs, target_a, target_b = model(inputs, targets, mixup_hidden= True, mixup_alpha = params.alpha, lam = lam)
            loss = mixup_criterion(criterion, outputs, target_a, target_b, lam)
            train_loss += loss.data.item()
            optimizer.zero_grad()
            loss.backward()
            
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (lam * predicted.eq(target_a.data).cpu().sum().float()
                        + (1 - lam) * predicted.eq(target_b.data).cpu().sum().float())
            
            bs = inputs.size(0)
            inputs_, targets_, a_ = [], [], []
            indices = np.arange(bs)
            np.random.shuffle(indices)
            
            split_size = int(bs/4)
            for j in indices[0:split_size]:
                x90 = inputs[j].transpose(2,1).flip(1)
                x180 = x90.transpose(2,1).flip(1)
                x270 = x180.transpose(2,1).flip(1)
                inputs_ += [inputs[j], x90, x180, x270]
                targets_ += [targets[j] for _ in range(4)]
                a_ += [torch.tensor(0),torch.tensor(1),torch.tensor(2),torch.tensor(3)]

            inputs = Variable(torch.stack(inputs_,0))
            targets = Variable(torch.stack(targets_,0))
            a_ = Variable(torch.stack(a_,0))

            inputs, targets, a_ = inputs.to(device), targets.to(device), a_.to(device)
            
            rf, outputs = model(inputs)
            rotate_outputs = rotate_classifier(rf)
            rloss = criterion(rotate_outputs, a_)
            closs = criterion(outputs, targets)

            loss = (rloss+closs)/2.0
            rotate_loss += rloss.data.item()
            loss.backward()
            optimizer.step()
            
            if (batch_idx + 1) % 50 == 0:
                print('{0}/{1}'.format(batch_idx,len(base_loader)), 
                             'Loss: %.3f | Acc: %.3f%% | RotLoss: %.3f  '
                             % (train_loss/(batch_idx+1),
                                100.*correct/total,rotate_loss/(batch_idx+1)))

        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            if not os.path.isdir(params.checkpoint_dir):
                os.makedirs(params.checkpoint_dir)

            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict() }, outfile)

        test_s2m2(base_loader_test, model, criterion)  

    return model 

if __name__ == '__main__':
    params = parse_args('train')
    base_file = data_dir[params.dataset] + 'base.json'
    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(save_dir, params.dataset, params.model, params.method)

    # DataLoader for train and test
    base_datamgr = SimpleDataManager(image_size, batch_size= params.batch_size)
    base_loader = base_datamgr.get_data_loader(base_file, aug= params.train_aug)
    base_datamgr_test = SimpleDataManager(image_size, batch_size= params.test_batch_size)
    base_loader_test = base_datamgr_test.get_data_loader(base_file, aug= False)

    model = wrn28_10(num_classes= params.num_classes)
    model = train_s2m2(base_loader, base_loader_test,  model, params, {})


   
   



  
