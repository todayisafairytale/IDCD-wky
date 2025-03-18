import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm


class _Unifier:
    @staticmethod
    def train(datahub, set_type, extractor=None, inter_func=None, **kwargs):
        if isinstance(inter_func, nn.Module):
            dataloader = datahub.to_dataloader(
                batch_size=kwargs["batch_size"],
                dtype=extractor.dtype,
                set_type=set_type,
                label=True
            )
            loss_func = kwargs["loss_func"]
            optimizer = kwargs["optimizer"]
            device = extractor.device
            epoch_losses = []
            extractor.train()
            inter_func.train()
            r_matrix:torch.Tensor=datahub.r_matrix(set_type="all")
            for batch_data in tqdm(dataloader, "Training"):
                student_id, exercise_id,q_mask,r= batch_data
                student_id: torch.Tensor = student_id.to(device)
                exercise_id: torch.Tensor = exercise_id.to(device)
                q_mask: torch.Tensor = q_mask.to(device)
                r:torch.Tensor=r.to(device)
                _ = extractor.extract(student_id,exercise_id,r_matrix)
                student_ts, exercise_ts= _[:2]
                compute_params = {
                    'student_ts': student_ts,
                    'exercise_ts':exercise_ts,
                    'q_mask': q_mask,
                }
                if len(_) > 2:
                    compute_params['other'] = _[2]
                    if isinstance(_[2], dict):
                        extra_loss = _[2].get('extra_loss', 0)
                else:
                    extra_loss = 0
                pred_r: torch.Tensor = inter_func.compute(**compute_params)
                loss = loss_func(pred_r, r.unsqueeze(1)) + extra_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                extractor.monotonicity()
                inter_func.monotonicity()
                epoch_losses.append(loss.mean().item())
            print("Average loss: {}".format(float(np.mean(epoch_losses))))
        # To cope with statistics methods
        else:
            ...

    @staticmethod
    def predict(datahub, set_type, extractor=None, inter_func=None, **kwargs):
        if isinstance(inter_func, nn.Module):
            dataloader = datahub.to_dataloader(
                batch_size=kwargs["batch_size"],
                dtype=extractor.dtype,
                set_type=set_type,
                label=False
            )
            device = extractor.device
            extractor.eval()
            inter_func.eval()
            pred = []
            r_matrix:torch.Tensor=datahub.r_matrix(set_type="all")
            for batch_data in tqdm(dataloader, "Evaluating"):
                student_id, exercise_id, q_mask = batch_data
                student_id: torch.Tensor = student_id.to(device)
                exercise_id: torch.Tensor = exercise_id.to(device)
                q_mask: torch.Tensor = q_mask.to(device)
                _ = extractor.extract(student_id,exercise_id, r_matrix)
                student_ts, exercise_ts = _[:2]
                extractor.mastery[student_id]=student_ts
                compute_params = {
                    'student_ts': student_ts,
                    'exercise_ts':exercise_ts,
                    'q_mask': q_mask,
                }
                if len(_) > 2:
                    compute_params['other'] = _[2]
                pred_r: torch.Tensor = inter_func.compute(**compute_params)
                pred.extend(pred_r.detach().cpu().tolist())
            return pred
        # To cope with statistics methods
        else:
            ...
