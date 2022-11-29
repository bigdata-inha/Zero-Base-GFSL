import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import utils
from .models import register


@register('fine-tune')
class FineTune(nn.Module):

    def __init__(self, encoder, classifier_args, classifier, encoder_args={}, method='cos',
                 temp=10., temp_learnable=True,):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        #self.encoder = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=64)
        self.method = method
        classifier_args['in_dim'] = self.encoder.out_dim
        self.classifier = models.make(classifier, **classifier_args)
        #추가 new classifier
        #self.classifier = nn.Linear(classifier_args['in_dim'],5)
        if temp_learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp


    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x

    '''def forward(self, x_shot, x_query):
        shot_shape = x_shot.shape[:-3]
        query_shape = x_query.shape[:-3]
        img_shape = x_shot.shape[-3:]

        x_shot = x_shot.view(-1, *img_shape)
        x_query = x_query.view(-1, *img_shape)
        x_tot = self.encoder(torch.cat([x_shot, x_query], dim=0))
        x_shot, x_query = x_tot[:len(x_shot)], x_tot[-len(x_query):]
        x_shot = x_shot.view(*shot_shape, -1)
        x_query = x_query.view(*query_shape, -1)

        if self.method == 'cos':
            x_shot = x_shot.mean(dim=-2)
            x_shot = F.normalize(x_shot, dim=-1)
            x_query = F.normalize(x_query, dim=-1)
            metric = 'dot'
        elif self.method == 'sqr':
            x_shot = x_shot.mean(dim=-2)
            metric = 'sqr'
        
        logits = utils.compute_logits(
                x_query, x_shot, metric=metric, temp=self.temp)
        return logits'''

