import argparse
from sched import scheduler
import yaml
import os
os.chdir('../')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from functools import reduce
from operator import mul

import datasets
from datasets import mini_imagenet
import models
from models.classifier import AdaptiveBiasClassifier, AdaptiveCenterClassifier, AdaptiveClassifier, LinearClassifier, centerClassifier, cosineClassifier, AlignClassifier
from models.rfs_resnet import   resnet12, weight_align, weight_normalize, adaptive_weight_align, adaptive_weight_normalize
import utils
import utils.few_shot as fs
from datasets.samplers import CategoriesSampler, DatasetSplit_tensor
import copy
import random

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return h


def main(config):
    # dataset
    novel_dataset = datasets.make(config['dataset'], **config['dataset_args'])
    utils.log('dataset: {} (x{}), {}'.format(
            novel_dataset[0][0].shape, len(novel_dataset), novel_dataset.n_classes))

    n_way = 5
 
    n_shot, n_query = args.shot, 15
    n_batch = 200
    ep_per_batch = 1
    
    if 'mini' in args.config:
        n_classes = 64
    elif 'tiered' in args.config:
        n_classes = 200
    else:
        n_classes = 800
    
    
    batch_sampler = CategoriesSampler(
            novel_dataset.label, n_batch, n_way, n_shot + n_query,
            ep_per_batch=ep_per_batch)
    novel_loader = DataLoader(novel_dataset, batch_sampler=batch_sampler,
                          num_workers=8, pin_memory=True)
    #using base data -GFSL
    #non_iid = args.non_iid
    #base_dataset = datasets.make(config['base_dataset'],
    #                            **config['base_dataset_args']) 
    #batch_sampler = CategoriesSampler(
    #        base_dataset.label, n_batch, non_iid, n_shot,
    #        ep_per_batch=1)
    
    
    #base_loader = DataLoader(base_dataset, batch_sampler=batch_sampler,
    #                    num_workers=8, pin_memory=True)
    
    
    val_dataset = datasets.make(config['val_dataset'],
                                **config['val_dataset_args'])
    val_loader = DataLoader(val_dataset, config['batch_size'],
                              num_workers=8, pin_memory=True)
    # model
    if config.get('load') is None:
        model = models.make('meta-baseline', encoder=None)
    else:
        model_sv = torch.load(config['load'])
        model_sv['model'] = 'fine-tune'
        model = models.make(model_sv['model'], **model_sv['model' + '_args'])
        
        # load pre-trained model
        encoder = models.load(torch.load(config['load'])).encoder
        classifier = models.load(torch.load(config['load'])).classifier
        model.encoder.load_state_dict(encoder.state_dict())
        model.classifier.load_state_dict(classifier.state_dict())


    if config.get('load_encoder') is not None:
        encoder = models.load(torch.load(config['load_encoder'])).encoder
        model.encoder = encoder

    if config.get('_parallel'):
        model = nn.DataParallel(model)

    model.train()
    utils.log('num params: {}'.format(utils.compute_n_params(model)))

    aves_keys = ['vl', 'va', 'vl_base', 'va_base','va_all','va_base_tmp']
    aves = {k: utils.Averager() for k in aves_keys}

    optimizer, lr_scheduler = utils.make_optimizer(
            model.parameters(),
            config['optimizer'], **config['optimizer_args'])
    
    batch = 10000

    test_epochs = args.test_epochs
    novel_epoch = args.novel_epochs
    
    # fix seed 
    seed = 0
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    
    va_lst = []
    va_base_lst = []
    va_all_lst = []
    
    # freeze bn
    if config.get('freeze_bn'):
        utils.freeze_bn(model) 
    # freeze extractor
    for name, param in model.named_parameters():
        if 'classifier' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    model.cuda()
    print(model)
    
    linear_weight = model.classifier.linear.weight.data.detach()
    linear_bias = model.classifier.linear.bias.data.detach()
    
    for epoch in range(1, test_epochs + 1):
        for batch_idx, novel_batch in (enumerate(novel_loader)):
            data, labels,data_idx = novel_batch
            
            data_idx = data_idx.view(n_way, n_shot + n_query)
            shot_idx, query_idx = data_idx.split([n_shot, n_query],dim=1)


            novel_dataset.fine_tune = True

            novel_few_loader = DataLoader(DatasetSplit_tensor(novel_dataset, shot_idx.reshape([n_way * n_shot])), n_batch, shuffle=False)
            
            
            x_shot, x_query = fs.split_shot_query(
                    data.cuda(), n_way, n_shot, n_query,
                    ep_per_batch=ep_per_batch)

            shot_shape = x_shot.shape[:-3]
            query_shape = x_query.shape[:-3]
            img_shape = x_shot.shape[-3:]
            x_shot = x_shot.view(-1, *img_shape)
            x_query = x_query.view(-1, *img_shape)
            # train

            model.classifier = AdaptiveCenterClassifier(640, n_classes, n_way, bias=True).cuda()
            model.classifier.base_linear.weight.data[:n_classes,:] = copy.deepcopy(linear_weight[:n_classes,:].detach())
            model.classifier.base_linear.bias.data[:n_classes] = copy.deepcopy(linear_bias[:n_classes].detach())
    
            model.cuda()
            model.train()
            if config.get('freeze_bn'):
                utils.freeze_bn(model) 
            optimizer = torch.optim.SGD([{'params': model.classifier.base_linear.parameters(), 'lr':args.lr},  
                                        {'params': model.classifier.novel_linear.parameters(),'lr':args.lr}],  
                                        lr=0.01, weight_decay=args.wd, momentum=0.9)

            label = fs.make_nk_label(n_way, n_shot,
                    ep_per_batch=ep_per_batch, n_classes=n_classes).cuda()
            #print ("Base mean:{}, Novel mean:{}".format( torch.mean(torch.mean(model.classifier.base_linear.weight.data,dim=1,keepdim=True)), torch.mean(torch.mean(model.classifier.novel_linear.weight.data,dim=1,keepdim=True))))
            #print ("Base var:{}, Novel var:{}".format( model.classifier.base_linear.weight.data.var(dim=1,keepdim=True).mean(), model.classifier.novel_linear.weight.data.var(dim=1,keepdim=True).mean()))
            
 
      
            for idx in range(1, novel_epoch+1):
                for novel_data, novel_label in (novel_few_loader):
                    novel_data, novel_label = novel_data.cuda(), novel_label.cuda()

                    novel_data.requires_grad=True
                    novel_label.requires_grad=False
                    
                    #only new
                    #logits = model(x_shot).view(-1, n_way)

                    logits = model(novel_data).view(-1, n_classes + n_way)
                    loss = F.cross_entropy(logits, label)
                    acc = utils.compute_acc(logits, label)


                    optimizer.zero_grad()
                    loss.backward()
                    #max_norm = 0.25
                    #torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), max_norm)
                    
                    optimizer.step()

                    logits = None; loss = None 
            
            classifier = model.classifier
            model.classifier = AdaptiveBiasClassifier(640, n_classes, n_way, bias=True).cuda()
            model.classifier.base_linear.weight.data = copy.deepcopy(classifier.base_linear.weight.data.detach())
            model.classifier.novel_linear.weight.data = copy.deepcopy(classifier.novel_linear.weight.data.detach())
            model.classifier.base_linear.bias.data = copy.deepcopy(classifier.base_linear.bias.data .detach())
            model.classifier.novel_linear.bias.data = copy.deepcopy(classifier.novel_linear.bias.data.detach())
            model.classifier.bias_linear.alpha.data[:n_classes] = (torch.std(model.classifier.novel_linear.weight.data,dim=1,keepdim=True).mean()\
                                                                   /  torch.std(model.classifier.base_linear.weight.data,dim=1,keepdim=True)).reshape(model.classifier.bias_linear.alpha.data[:n_classes].shape)

            model.cuda()
            model.train()
            if config.get('freeze_bn'):
                utils.freeze_bn(model) 
            
            optimizer_balanced = torch.optim.SGD([{'params': model.classifier.base_linear.parameters(), 'lr':0.000},  
                                        {'params': model.classifier.novel_linear.parameters(),'lr':0.00},
                                        {'params': model.classifier.bias_linear.parameters(),'lr':args.add_lr}],  
                                         weight_decay=5e-4,momentum=0.9)

            label = fs.make_nk_label(n_way, n_shot,
                    ep_per_batch=ep_per_batch, n_classes=n_classes).cuda()
            #print ("Base mean:{}, Novel mean:{}".format( torch.mean(torch.mean(model.classifier.base_linear.weight.data,dim=1,keepdim=True)), torch.mean(torch.mean(model.classifier.novel_linear.weight.data,dim=1,keepdim=True))))
            #print ("Base var:{}, Novel var:{}".format( model.classifier.base_linear.weight.data.var(dim=1,keepdim=True).mean(), model.classifier.novel_linear.weight.data.var(dim=1,keepdim=True).mean()))
            
            
            for idx in range(1, args.add_iter+1):
                for ovel_data, novel_label  in (novel_few_loader):
                    novel_data, novel_label = novel_data.cuda(), novel_label.cuda()

                    novel_data.requires_grad=True
                    novel_label.requires_grad=False
                    #only new
                    #logits = model(x_shot).view(-1, n_way)
    

                    logits = model(novel_data).view(-1, n_classes + n_way)
                    
                    loss = F.cross_entropy(logits, label)
                    acc = utils.compute_acc(logits, label)


                    optimizer_balanced.zero_grad()
                    loss.backward()
                    #max_norm = 0.25
                    #torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), max_norm)
                    
                    optimizer_balanced.step()
                    #scheduler.step()

                    logits = None; loss = None 
                    #model.classifier.linear.weight.data[:n_classes,:] = copy.deepcopy(linear_weight[:n_classes,:].detach())

            #print ("Base mean:{}, Novel mean:{}".format( torch.mean(torch.mean(model.classifier.base_linear.weight.data,dim=1,keepdim=True)), torch.mean(torch.mean(model.classifier.novel_linear.weight.data,dim=1,keepdim=True))))
           
            #print ("Base var:{}, Novel var:{}".format( model.classifier.base_linear.weight.data.var(dim=1,keepdim=True).mean(), model.classifier.novel_linear.weight.data.var(dim=1,keepdim=True).mean()))


            # test
            novel_dataset.fine_tune = False
            model.eval()
            with torch.no_grad():
                 
                #mean_base = torch.mean(torch.norm(model.classifier.base_linear.weight.data,dim=1)).item()
                #mean_novel = torch.mean(torch.norm(model.classifier.novel_linear.weight.data,dim=1)).item()
                #print("mean_base:{}, mean_novel:{}".format(mean_base, mean_novel))

                logits = model(x_query).view(-1, n_classes + n_way)
                label = fs.make_nk_label(n_way, n_query,
                        ep_per_batch=ep_per_batch, n_classes=n_classes).cuda()
                loss = F.cross_entropy(logits, label)
                acc = utils.compute_acc(logits, label)

                aves['vl'].add(loss.item(), len(data))
                aves['va'].add(acc, len(data))
                va_lst.append(acc)
                
                #base
                for data_base, label_base,_ in (val_loader):
                    data_base, label_base = data_base.cuda(), label_base.cuda()
                    with torch.no_grad():
                        logits_base = model(data_base)
                        lose_base = F.cross_entropy(logits_base, label_base)
                        acc_base = utils.compute_acc(logits_base, label_base)

                    aves['vl_base'].add(lose_base.item())
                    aves['va_base'].add(acc_base)
                    va_base_lst.append(acc_base)
                    aves['va_base_tmp'].add(acc_base)
                    va_base_lst.append(acc_base)
               
                aves['va_all'].add((acc+aves['va_base_tmp'].item())/2)
                va_all_lst.append((acc+aves['va_base_tmp'].item())/2)
                aves['va_base_tmp'].n = 0
                aves['va_base_tmp'].v = 0
                print('\n epoch:{} novel:{:.2f} | {:.2f}, base:{:.2f}, all:{:.2f}'.format(batch_idx, acc*100, aves['va'].item() * 100, aves['va_base'].item() * 100, aves['va_all'].item() * 100))
        print('test epoch {}: acc={:.2f} +- {:.2f} (%), loss={:.4f} (@{})'.format(
                epoch, aves['va'].item() * 100,
                mean_confidence_interval(va_lst) * 100,
                aves['vl'].item(), labels[-1]))
        print(' val base {:.4f}|{:.4f} +- {:.2f} (%) '.format(aves['vl_base'].item(), aves['va_base'].item()*100, mean_confidence_interval(va_base_lst) * 100 ))
        print(' all base {:.4f}|{:.4f} +- {:.2f} (%) '.format(aves['vl_base'].item(), aves['va_all'].item()*100, mean_confidence_interval(va_all_lst) * 100 ))

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/test_few_shot_tiered.yaml')
parser.add_argument('--shot', type=int, default=5)
parser.add_argument('--test-epochs', type=int, default=10)
parser.add_argument('--sauc', action='store_true')
parser.add_argument('--gpu', default='0')
parser.add_argument('--wd', default=1e-2)
parser.add_argument('--lr', default=0.001)
parser.add_argument('--novel-epochs', default=100)
parser.add_argument('--add_iter', default=500)
parser.add_argument('--add_lr', default=0.001)
# args = parser.parse_args()
'''
args = easydict.EasyDict({
    'config': 'configs/test_few_shot_mini.yaml',
    'test_epochs': 3,
    'sauc': False,
    'gpu': '0',
    'lr': 0.001,
    'wd': 5e-4,
    'shot': 5,
    'novel_epochs':500,
    'add_iter': 50,
    'add_lr': 0.001,

})
'''
config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
if len(args.gpu.split(',')) > 1:
    config['_parallel'] = True

utils.set_gpu(args.gpu)
main(config)
