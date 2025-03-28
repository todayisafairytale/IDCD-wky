import torch
import torch.nn as nn
import torch.optim as optim

from .._base import _CognitiveDiagnosisModel
from ..datahub import DataHub
from ..interfunc import IDCD_IF
from ..extractor import IDCD_EX


class IDCDM(_CognitiveDiagnosisModel):
    def __init__(self, student_num: int, exercise_num: int, knowledge_num: int,stu_latent_dim,exer_latent_dim,save_flag=False):
        super().__init__(student_num, exercise_num, knowledge_num)
        self.stu_latent_dim=stu_latent_dim
        self.exer_latent_dim=exer_latent_dim

    def build(self, hidden_dims: list = None, dropout=0.5, device="cpu", dtype=torch.float32, **kwargs):

        self.extractor = IDCD_EX(
            student_num=self.student_num,
            exercise_num=self.exercise_num,
            knowledge_num=self.knowledge_num,
            device=device,
            dtype=dtype
        )
       
        self.inter_func = IDCD_IF(knowledge_num=self.knowledge_num,
                                 stu_latent_dim=self.stu_latent_dim,
                                 exer_latent_dim=self.exer_latent_dim,
                                 dropout=dropout,
                                 device=device,
                                 dtype=dtype)
        
    def train(self, datahub: DataHub, set_type="train", valid_set_type="valid",
              valid_metrics=None, epoch=10, lr=2e-3,  batch_size=256):
        if valid_metrics is None:
            valid_metrics = ['auc','acc','doa']
        loss_func = nn.BCELoss()
        optimizer = optim.Adam([{'params': self.extractor.parameters(),
                                 'lr': lr},
                                {'params': self.inter_func.parameters(),
                                 'lr': lr}])
        for epoch_i in range(0, epoch):
            print("[Epoch {}]".format(epoch_i + 1))
            self._train(datahub=datahub, set_type=set_type,
                        valid_set_type=valid_set_type, valid_metrics=valid_metrics,
                        batch_size=batch_size, loss_func=loss_func, optimizer=optimizer)
        if self.save_flag:
            self.save("save.pth")
    
    def predict(self, datahub: DataHub, set_type, batch_size=256, **kwargs):
        return self._predict(datahub=datahub, set_type=set_type, batch_size=batch_size)

    def score(self, datahub: DataHub, set_type, metrics: list, batch_size=256, **kwargs) -> dict:
        if metrics is None:
            metrics = ["auc","acc","doa"]
        return self._score(datahub=datahub, set_type=set_type, metrics=metrics, batch_size=batch_size)

    def diagnose(self):
        if self.inter_func is Ellipsis or self.extractor is Ellipsis:
            raise RuntimeError("Call \"build\" method to build interaction function before calling this method.")
        return self.inter_func.transform(self.extractor["mastery"])

    def load(self, ex_path: str, if_path: str):
        if self.inter_func is Ellipsis or self.extractor is Ellipsis:
            raise RuntimeError("Call \"build\" method to build interaction function before calling this method.")
        self.extractor.load_state_dict(torch.load(ex_path))
        self.inter_func.load_state_dict(torch.load(if_path))

    def save(self, ex_path: str, if_path: str):
        if self.inter_func is Ellipsis or self.extractor is Ellipsis:
            raise RuntimeError("Call \"build\" method to build interaction function before calling this method.")
        torch.save(self.extractor.state_dict(), ex_path)
        torch.save(self.inter_func.state_dict(), if_path)

    def get_attribute(self, attribute_name):
        if attribute_name == 'mastery':
            return self.diagnose().detach().cpu().numpy()
        elif attribute_name == 'diff':
            return self.inter_func.transform(self.extractor["diff"]).detach().cpu().numpy()
        elif attribute_name == 'knowledge':
            return self.extractor["knowledge"].detach().cpu().numpy()
        elif attribute_name == 'over_estimate':
            return self.extractor["over_estimate"].detach().cpu().numpy()
        else:
            return None
