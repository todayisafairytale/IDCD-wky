import torch
import numpy as np
import argparse
import sys
import os
import wandb as wb
from pprint import pprint
from inscd import listener
from inscd.datahub import DataHub
from inscd.models import IDCDM


def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--method', default='idcdm', type=str,
                    help='method')
parser.add_argument('--epoch', type=int, help='epoch of method', default=10)
parser.add_argument('--seed', default=0, type=int, help='seed for exp')
parser.add_argument('--data_type',default='SLP',help='dataset')
parser.add_argument('--dtype', default=torch.float64, help='dtype of tensor')
parser.add_argument('--device', default='cpu', type=str, help='device for exp')
parser.add_argument('--batch_size', type=int, help='batch size of benchmark', default=256)
parser.add_argument('--lr', type=float, help='learning rate', default=7e-3)
parser.add_argument('--test_size',default=0.2,help='test size')
parser.add_argument('--stu_latent_dim',default=32,help='student latent dimension')
parser.add_argument('--exer_latent_dim',default=32,help='exercise latent dimension')

config_dict = vars(parser.parse_args())

method_name = config_dict['method']
name = f"seed{config_dict['seed']}--nonpos--lr={config_dict['lr']}"
tags = [config_dict['method'], str(config_dict['seed'])]
config_dict['name'] = name
method = config_dict['method']
datatype = config_dict['data_type']

pprint(config_dict)
run = wb.init(project="idcd", name=name,
              tags=tags,
              config=config_dict)
config_dict['id'] = run.id


def main(config):
    def print_plus(tmp_dict, if_wandb=True):
        pprint(tmp_dict)
        if if_wandb:
            wb.log(tmp_dict)

    listener.update(print_plus)
    set_seeds(config['seed'])
    datahub = DataHub(f"datasets/{config['data_type']}")
    metrics = ['auc','acc','doa'] 
    print("Number of train_response logs {}".format(len(datahub)))
    idcdm = IDCDM(datahub.student_num, datahub.exercise_num, datahub.knowledge_num,stu_latent_dim=config['stu_latent_dim'],exer_latent_dim=config['exer_latent_dim'])
    idcdm.build(device=config['device'])
    idcdm.train(datahub, "train", "test", valid_metrics=metrics, batch_size=config['batch_size'],epoch=config['epoch'], lr=config['lr'])
if __name__ == '__main__':
    sys.exit(main(config_dict))
