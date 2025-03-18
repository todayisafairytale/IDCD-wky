import torch
import torch.nn as nn
from collections import OrderedDict
from .._base import _Extractor
from ..interfunc._util import none_neg_clipper


class IDCD_EX(_Extractor, nn.Module):
    def __init__(self, student_num: int, exercise_num: int, knowledge_num: int, device, dtype):
        super().__init__()
        self.student_num = student_num
        self.exercise_num = exercise_num
        self.knowledge_num = knowledge_num
        self.device = device
        self.dtype = dtype
        self.mastery=torch.Tensor(torch.zeros(size=(student_num,knowledge_num),requires_grad=False))
        
        self.__student_emb = nn.Sequential(
            OrderedDict(
                [
                    ('linear_1', nn.Linear(exercise_num, 256)),
                    ('activate_1', nn.Sigmoid()),
                    ('kinear_2', nn.Linear(256, knowledge_num)),
                    ('activate_2', nn.Sigmoid()),
                ]
            )
        ).to(device)
        self.__exercise_emb = nn.Sequential(
            OrderedDict(
                [
                    ('linear_1', nn.Linear(student_num, 512)),
                    ('activate_1', nn.Sigmoid()),
                    ('kinear_2', nn.Linear(512, 256)),
                    ('activate_2', nn.Sigmoid()),
                    ('linear_last',nn.Linear(256,knowledge_num)),
                    ('activate_3',nn.Sigmoid())
                ]
            )
        ).to(device)
        self.apply(self.initialize_weights)
        self.__emb_map={}

    @staticmethod
    def initialize_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_normal_(module.weight)
   
    def monotonicity(self): 
        for layer in self.__student_emb:
            if isinstance(layer, nn.Linear):
                layer.apply(none_neg_clipper)

    def extract(self, student_id, exercise_id, r_matrix):
        student_ts = self.__student_emb(torch.Tensor(r_matrix[student_id]))
        exercise_ts=self.__exercise_emb(torch.Tensor(r_matrix.T[exercise_id]))
        return student_ts, exercise_ts
    def __getitem__(self, item):
        self.__emb_map["mastery"] = self.mastery
        return self.__emb_map[item]

