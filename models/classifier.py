import math

import torch
import torch.nn as nn

import models
from models.rfs_resnet import  CosineLinear, ZeroMean_Classifier, BiasLayer,ZeroMeanCosineLinear,NormalizeLinear
import utils
from .models import register


@register('classifier')
class Classifier(nn.Module):
    
    def __init__(self, encoder, encoder_args,
                 classifier, classifier_args):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        classifier_args['in_dim'] = self.encoder.out_dim
        self.classifier = models.make(classifier, **classifier_args)

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x
    
@register('adaptive-classifier')
class AdaptiveClassifier(nn.Module):

    def __init__(self, in_dim, base_classes, novel_classes, bias=True):
        super().__init__()
        self.base_linear = nn.Linear(in_dim, base_classes,bias)
        self.novel_linear = nn.Linear(in_dim, novel_classes,bias)
        
    def forward(self, x):
        #print(self.linear.weight.data.mean(dim=1,keepdim=True))
        base_output = self.base_linear(x)
        novel_output = self.novel_linear(x)
        
        total_ouput = torch.cat([base_output, novel_output], dim=1)

        return total_ouput
    
@register('adaptiveCosine-classifier')
class AdaptiveCosineClassifier(nn.Module):

    def __init__(self, in_dim, base_classes, novel_classes):
        super().__init__()
        self.base_linear = NormalizeLinear(in_dim, base_classes,bias)
        self.novel_linear = NormalizeLinear(in_dim, novel_classes,bias)
        
    def forward(self, x):
        #print(self.linear.weight.data.mean(dim=1,keepdim=True))
        base_output = self.base_linear(x)
        novel_output = self.novel_linear(x)
        
        total_ouput = torch.cat([base_output, novel_output], dim=1)

        return total_ouput
    
        
@register('bias-classifier')
class AdaptiveBiasClassifier(nn.Module):

    def __init__(self, in_dim, base_classes, novel_classes, bias=True):
        super().__init__()
        self.base_linear = nn.Linear(in_dim, base_classes, bias)
        self.novel_linear =  ZeroMean_Classifier(in_dim, novel_classes, bias)
        self.bias_linear = BiasLayer(base_classes + novel_classes)
        
    def forward(self, x):
        #print(self.linear.weight.data.mean(dim=1,keepdim=True))
        base_output = (self.base_linear(x))
        novel_output = (self.novel_linear(x))        
        
        total_ouput = torch.cat([base_output, novel_output], dim=1)

        return self.bias_linear(total_ouput)
    
@register('align-classifier')
class AlignClassifier(nn.Module):

    def __init__(self, in_dim, base_classes, novel_classes, bias=True):
        super().__init__()
        self.base_linear = nn.Linear(in_dim, base_classes, bias)
        self.novel_linear = nn.Linear(in_dim, novel_classes, bias)
        self.base_sigma = nn.Parameter(torch.Tensor(base_classes))
        self.novel_sigma = nn.Parameter(torch.Tensor(novel_classes)) 
   
    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
  
        mean_base = torch.mean(torch.norm(self.base_linear.weight.data,dim=1)).item()
        mean_novel = torch.mean(torch.norm(self.novel_linear.weight.data,dim=1)).item()
        gamma = mean_base / mean_novel
        self.novel_sigma.data.fill_(100) 
        self.base_sigma.data.fill_(1) 
        print(gamma)
    def forward(self, x):
        #print(self.linear.weight.data.mean(dim=1,keepdim=True))
        base_output = self.base_sigma * self.base_linear(x)
        novel_output =self.novel_sigma * self.novel_linear(x)
        
        total_ouput = torch.cat([base_output, novel_output], dim=1)

        return total_ouput
@register('linear-classifier')
class LinearClassifier(nn.Module):

    def __init__(self, in_dim, n_classes, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes, bias)
        
    def forward(self, x):
        #print(self.linear.weight.data.mean(dim=1,keepdim=True))
        return self.linear(x)

@register('center-classifier')
class centerClassifier(nn.Module):

    def __init__(self, in_dim, n_classes, bias=True):
        super().__init__()
        self.linear = ZeroMean_Classifier(in_dim, n_classes, bias)
        
    def forward(self, x):
        #print(self.linear.weight.data.mean(dim=1,keepdim=True))
        return self.linear(x)

@register('adaptiveCenter-classifier')
class AdaptiveCenterClassifier(nn.Module):

    def __init__(self, in_dim, base_classes, novel_classes, bias=True):
        super().__init__()
        self.base_linear = nn.Linear(in_dim, base_classes, bias)
        self.novel_linear = ZeroMean_Classifier(in_dim, novel_classes, bias)
        
    def forward(self, x):
        #print(self.linear.weight.data.mean(dim=1,keepdim=True))
        base_output = self.base_linear(x)
        novel_output = self.novel_linear(x)
        
        total_ouput = torch.cat([base_output, novel_output], dim=1)

        return total_ouput

@register('cosine-classifier')
class cosineClassifier(nn.Module):

    def __init__(self, in_dim, n_classes, bias=False):
        super().__init__()
        self.linear = CosineLinear(in_dim, n_classes)
        
    def forward(self, x):
        #print(self.linear.weight.data.mean(dim=1,keepdim=True))
        return self.linear(x)

@register('nn-classifier')
class NNClassifier(nn.Module):

    def __init__(self, in_dim, n_classes, metric='cos', temp=None):
        super().__init__()
        self.proto = nn.Parameter(torch.empty(n_classes, in_dim))
        nn.init.kaiming_uniform_(self.proto, a=math.sqrt(5))
        if temp is None:
            if metric == 'cos':
                temp = nn.Parameter(torch.tensor(10.))
            else:
                temp = 1.0
        self.metric = metric
        self.temp = temp

    def forward(self, x):
        return utils.compute_logits(x, self.proto, self.metric, self.temp)

