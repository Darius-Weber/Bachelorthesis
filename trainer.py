from functools import partial

import numpy as np
import torch
from torch_scatter import scatter


class Trainer:
    def __init__(self,
                 device,
                 loss_target,
                 loss_type,
                 micro_batch,
                 ipm_steps,
                 ipm_alpha,
                 loss_weight):
        assert 0. <= ipm_alpha <= 1.
        self.ipm_steps = ipm_steps
        self.step_weight = torch.tensor([ipm_alpha ** (ipm_steps - l - 1)
                                         for l in range(ipm_steps)],
                                        dtype=torch.float, device=device)[None]
        # self.best_val_loss = 1.e8
        self.best_val_objgap = 100.
        self.best_val_consgap = 100.
        self.patience = 0
        self.device = device
        self.loss_target = loss_target.split('+')
        self.loss_weight = loss_weight
        if loss_type == 'l2':
            self.loss_func = partial(torch.pow, exponent=2)
        elif loss_type == 'l1':
            self.loss_func = torch.abs
        else:
            raise ValueError
        self.micro_batch = micro_batch

    def train(self, dataloader, model, optimizer):
        model.train()
        optimizer.zero_grad()

        update_count = 0
        micro_batch = int(min(self.micro_batch, len(dataloader)))
        loss_scaling_lst = [micro_batch] * (len(dataloader) // micro_batch) + [len(dataloader) % micro_batch]

        train_losses = 0.
        num_graphs = 0
        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            vals = model(data)
            loss = self.get_loss(vals, data)

            train_losses += loss.detach() * data.num_graphs
            num_graphs += data.num_graphs

            update_count += 1
            loss = loss / float(loss_scaling_lst[0])  # scale the loss
            loss.backward()

            if update_count >= micro_batch or i == len(dataloader) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               max_norm=1.0,
                                               error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad()
                update_count = 0
                loss_scaling_lst.pop(0)

        return train_losses.item() / num_graphs

    @torch.no_grad()
    def eval(self, dataloader, model, scheduler=None):
        model.eval()

        val_losses = 0.
        num_graphs = 0
        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            vals = model(data)
            loss = self.get_loss(vals, data)
            val_losses += loss * data.num_graphs
            num_graphs += data.num_graphs
        val_loss = val_losses.item() / num_graphs

        if scheduler is not None:
            scheduler.step(val_loss)
        return val_loss

    def get_loss(self, vals, data):
        #vals predicted values
        #data.gt_primals (ground truth)
        loss = 0.

        #if 'obj' in self.loss_target:
        #    pred = vals[:, -self.ipm_steps:]
        #    c_times_x = data.obj_const[:, None] * pred
        #    obj_pred = scatter(c_times_x, data['vals'].batch, dim=0, reduce='sum')
        #    obj_pred = (self.loss_func(obj_pred) * self.step_weight).mean()
        #    loss = loss + obj_pred
        #don't needed?
        if 'primal' in self.loss_target:
            primal_loss = (self.loss_func(
                vals[:, -self.ipm_steps:] -
                data.gt_primals[:, -self.ipm_steps:]
            ) * self.step_weight).mean()
            loss = loss + primal_loss * self.loss_weight['primal']
        if 'objgap' in self.loss_target:
            obj_loss = (self.loss_func(
                self.get_obj_metric(data, vals, hard_non_negative=False)) * self.step_weight).mean()
            loss = loss + obj_loss * self.loss_weight['objgap']
        if 'constrain' in self.loss_target:
            constraint_gap_eq = self.get_constraint_violation_eq(vals, data)
            constraint_gap_uq = self.get_constraint_violation_uq(vals, data)
            cons_loss = (self.loss_func(constraint_gap_eq) * self.step_weight).mean() + (self.loss_func(constraint_gap_uq) * self.step_weight).mean()
            loss = loss + cons_loss * self.loss_weight['constraint']
        return loss

    def get_constraint_violation_uq(self, pred, data):
        """
        Gx - h
        :param pred:
        :param data:
        :return:
        """
        #CONSTRAIN VIOLTATION
        pred = pred[:, -self.ipm_steps:]
        Gx = scatter(pred[data.G_col, :] * data.G_val[:, None], data.G_row, reduce='sum', dim=0)

        constraint_gap = torch.relu(Gx - data.h[:, None])
        return constraint_gap

    def get_constraint_violation_eq(self, pred, data):
        """
        Ax - b
        :param pred:
        :param data:
        :return:
        """
        #CONSTRAIN VIOLTATION
        pred = pred[:, -self.ipm_steps:]
        Ax = scatter(pred[data.A_col, :] * data.A_val[:, None], data.A_row, reduce='sum', dim=0)

        constraint_gap = Ax - data.b[:, None]
        return constraint_gap

    def get_obj_metric(self, data, pred, hard_non_negative=False):
        # if hard_non_negative, we need a relu to make x all non-negative
        # just for metric usage, not for training
        #ABSOLUTE OBJECTIVE GAP
        # TODO: Look if it really does what it should!
        pred = pred[:, -self.ipm_steps:]
        if hard_non_negative:
            pred = torch.relu(pred)
        c_times_x = data.q[:, None] * pred  #q*x
        obj_pred_c = scatter(c_times_x, data['vals'].batch, dim=0, reduce='sum')
        x_gt = data.gt_primals[:, -self.ipm_steps:]
        c_times_xgt = data.q[:, None] * x_gt
        obj_gt_c = scatter(c_times_xgt, data['vals'].batch, dim=0, reduce='sum')
        #TODO: Look what is better
        #xQx_pred = (scatter(pred[data.Q_col, :] * data.Q_val[:, None] * pred[data.Q_row, :], data.Q_row, reduce='sum',
        #                    dim=0).sum()) / 2  #maybe not sum()/2?
        xQx_pred = torch.sum(pred[data.Q_col, :] * data.Q_val[:, None] * pred[data.Q_row, :], axis=0)
        #xQx_gt = (scatter(x_gt[data.Q_col, :] * data.Q_val[:, None] * x_gt[data.Q_row, :], data.Q_row, reduce='sum',
        #                    dim=0).sum()) / 2
        xQx_gt = torch.sum(pred[data.Q_col, :] * data.Q_val[:, None] * pred[data.Q_row, :], axis=0)
        obj_pred = obj_pred_c + xQx_pred
        obj_gt = obj_gt_c + xQx_gt
        #print("Obj metric groundtruth: ", obj_pred)
        #print("Obj metric groundtruth: ", obj_gt)
        return (obj_pred - obj_gt) / obj_gt

    def obj_metric(self, dataloader, model):
        model.eval
        obj_gap = []
        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            vals = model(data)
            obj_gap.append(np.abs(self.get_obj_metric(data, vals, hard_non_negative=True).detach().cpu().numpy()))

        return np.concatenate(obj_gap, axis=0)

    @torch.no_grad()
    def eval_metrics(self, dataloader, model):
        """
        both obj and constraint gap

        :param dataloader:
        :param model:
        :return:
        """
        model.eval()

        cons_gap_eq = []
        cons_gap_uq = []
        obj_gap = []
        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            vals = model(data)
            constrain_violation_eq = self.get_constraint_violation_eq(vals, data)
            constrain_violation_uq = self.get_constraint_violation_uq(vals, data)

            cons_gap_eq.append(np.abs(constrain_violation_eq).detach().cpu().numpy())
            cons_gap_uq.append(np.abs(constrain_violation_uq.detach().cpu().numpy()))
            obj_gap.append(np.abs(self.get_obj_metric(data, vals, hard_non_negative=True).detach().cpu().numpy()))

        obj_gap = np.concatenate(obj_gap, axis=0)
        cons_gap_eq = np.concatenate(cons_gap_eq, axis=0)
        cons_gap_uq = np.concatenate(cons_gap_uq, axis=0)
        return obj_gap, cons_gap_eq, cons_gap_uq
