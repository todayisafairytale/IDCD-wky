import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from .._base import _InteractionFunction


class IDCD_IF(_InteractionFunction, nn.Module):
    def __init__(self, knowledge_num: int, stu_latent_dim,exer_latent_dim,dropout, device, dtype):
        super().__init__()
        self.knowledge_num = knowledge_num
        self.stu_latent_dim=stu_latent_dim,
        self.exer_latent_dim=exer_latent_dim,
        self.dropout = dropout
        self.device = device
        self.dtype = dtype
        stu_layer=OrderedDict([
            ('stu_linear_1',nn.Linear(knowledge_num,stu_latent_dim)),
            ('activate_1',nn.Sigmoid())
        ])
        exer_layer=OrderedDict([
            ('exer_linear_1',nn.Linear(knowledge_num,exer_latent_dim)),
            ('activate_1',nn.Sigmoid())
        ])
        pred_layers = OrderedDict([
                ('pred_layer_1', nn.Linear(stu_latent_dim, 64)),
                ('pred_activate_1', nn.Sigmoid()),
                ('pred_dropout_1', nn.Dropout(p=0.5)),
                ('pred_layer_2', nn.Linear(64, 32)),
                ('pred_activate_2', nn.Sigmoid()),
                ('pred_dropout_2', nn.Dropout(p=0.5)),
                ('pred_layer_3', nn.Linear(32, 1)),
                ('pred_activate_3', nn.Sigmoid()),
            ])
        self.stu_L=nn.Sequential(stu_layer).to(device)
        self.exer_L=nn.Sequential(exer_layer).to(device)
        self.mlp = nn.Sequential(pred_layers).to(self.device)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        
    def compute(self, **kwargs):
        student_ts = kwargs["student_ts"]
        exercise_ts=kwargs["exercise_ts"]
        q_mask = kwargs["q_mask"]
        input_Mas = self.stu_L(torch.Tensor(student_ts*q_mask))
        input_diff=self.exer_L(torch.Tensor(exercise_ts*q_mask))
        return self.mlp(input_Mas-input_diff)

    def transform(self, tensor):
        return F.sigmoid(tensor)

