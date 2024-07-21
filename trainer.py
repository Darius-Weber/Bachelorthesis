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
        Gx = scatter(pred[data.G_col, :] * data.G_val[:, None], data.G_row, reduce='sum', dim=0, dim_size=data.h[:, None].shape[0])
        constraint_gap = torch.relu(Gx - data.h[:, None])
        # Normalize constraint_violation_uq
        # Logarithmic scaling for constraint_violation_uq
        #print("before log",constraint_gap)
        if constraint_gap.numel() > 0:
            log_scaled_gap = torch.log1p(torch.abs(constraint_gap))
        #    max_log_scaled_gap = log_scaled_gap.max()
        #    if max_log_scaled_gap > 0:
        #        log_scaled_gap = log_scaled_gap / max_log_scaled_gap.detach()
        #        print("logscaled", log_scaled_gap)
        #    print("after log", log_scaled_gap)
            return log_scaled_gap
        #print("constraint_gap",constraint_gap)
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
        #print("gap-eq",constraint_gap)
        # Normalize constraint_violation_eq
        # Logarithmic scaling for constraint_violation_eq
        #print("before log eq", constraint_gap)
        if constraint_gap.numel() > 0:
            log_scaled_gap = torch.log1p(torch.abs(constraint_gap))
        #    max_log_scaled_gap = log_scaled_gap.max()
        #    if max_log_scaled_gap > 0:
        #        log_scaled_gap = log_scaled_gap / max_log_scaled_gap.detach()
        #        print("logscaled", log_scaled_gap)
            #print("after log", log_scaled_gap)
            return log_scaled_gap
        #print("constraint_gap",constraint_gap)
        return constraint_gap

    def get_obj_metric(self, data, pred, hard_non_negative=False):
        # if hard_non_negative, we need a relu to make x all non-negative
        # just for metric usage, not for training

        pred = pred[:, -self.ipm_steps:]
        if hard_non_negative:
            pred = torch.relu(pred)
        c_times_x = data.q[:, None] * pred  #q*x
        obj_pred_c = scatter(c_times_x, data['vals'].batch, dim=0, reduce='sum')
        x_gt = data.gt_primals[:, -self.ipm_steps:]
        c_times_xgt = data.q[:, None] * x_gt
        obj_gt_c = scatter(c_times_xgt, data['vals'].batch, dim=0, reduce='sum')

        slice = data._slice_dict['Q_val']
        num_nonzero_Q = slice[1:] - slice[:-1]
        Q_batch = torch.arange(len(num_nonzero_Q)).repeat_interleave(num_nonzero_Q).to(self.device)
        xQx_pred = scatter(pred[data.Q_col, :] * data.Q_val[:, None] * pred[data.Q_row, :], Q_batch, reduce='sum', dim=0)

        xQx_gt = scatter(x_gt[data.Q_col, :] * data.Q_val[:, None] * x_gt[data.Q_row, :], Q_batch, reduce='sum', dim=0)

        #xQx_gt = scatter(xQx_gt_Q, data['vals'].batch, dim=0, reduce='sum')

        obj_pred = obj_pred_c + xQx_pred
        obj_gt = obj_gt_c + xQx_gt
        #print("obj_value",data.obj_value)
        #print("obj_gt_c", obj_gt_c)
        #print("obj_pred",obj_pred)
        #print("obj_gt",obj_gt)

        #print("obj_gap",(obj_pred - obj_gt)/(obj_gt+1e-5)) #This can be a really high value because obj_gt is sometimes very small (e.g. with softmargin svm). Therefore, there is also a high train loss.
        #print("obj_diff",(obj_pred - obj_gt)) #not normalized

        #Suggestions for a different normalization:
        #print("tanh", torch.tanh((obj_pred - obj_gt))) #Good for getting small train loss, but (high) difference is cut to 1, -1
        #print("log+1", torch.log1p(torch.abs(obj_pred - obj_gt))) #(high) difference is not cutted, but again leads to higher train loss, but not astronomically high (more between 0-100)
        #return (obj_pred - obj_gt) / (obj_gt + 1e-5)
        #print("log+1 scaled",torch.log1p(torch.abs(obj_pred - obj_gt))/ torch.log1p(torch.abs(obj_pred - obj_gt)).max().detach()) #(high) difference is not cut up, and train loss is in the range of (0-10) (own implementation)
        
        #obj_diff = obj_pred - obj_gt
        #max_obj_diff = obj_diff.abs().max()
        #if max_obj_diff > 0:
        #    obj_diff = obj_diff / max_obj_diff.detach()
        #return obj_diff
        #return torch.log1p(torch.abs(obj_pred - obj_gt))/ torch.log1p(torch.abs(obj_pred - obj_gt)).max().detach()#(obj_gt+1e-5)
        #obj_diff = obj_pred - obj_gt
        #print("before obj_diff")
        #obj_diff = obj_pred - obj_gt
        #print("obj diff", obj_diff)
        #obj_gap = obj_diff/(1+torch.abs(obj_gt)).detach()
        #print("denominator", (1+torch.abs(obj_gt)))
        #print("obj_gap", obj_gap)
        #log_scaled_diff = torch.log1p(torch.abs(obj_diff))
        #print("after obj_dif", log_scaled_diff)
        #max_log_scaled_diff = log_scaled_diff.max()
        #if max_log_scaled_diff > 0:
        #    log_scaled_diff = log_scaled_diff / max_log_scaled_diff.detach()
        # Calculate the difference
        #print("before normalization", torch.log1p(torch.abs(obj_pred - obj_gt)))
        #print("after", torch.log1p(torch.abs(obj_pred - obj_gt))/(torch.log1p(torch.abs(obj_pred - obj_gt)).max()+1))
        diff = obj_pred - obj_gt
        return torch.log1p(torch.abs(diff))
        #return torch.log1p(torch.abs(obj_pred - obj_gt))/(torch.log1p(torch.abs(obj_pred - obj_gt)).max()+1).detach()
        #--------------------------Important----------------------------------#
    def obj_metric(self, dataloader, model):
        model.eval
        obj_gap = []
        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            vals = model(data)
            obj_gap.append(np.abs(self.get_obj_metric(data, vals, hard_non_negative=False).detach().cpu().numpy())) #TODO: look for changes (default False)

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
            constraint_violation_eq = self.get_constraint_violation_eq(vals, data)
            constraint_violation_uq = self.get_constraint_violation_uq(vals, data)

            cons_gap_eq.append(np.abs(constraint_violation_eq.detach().cpu().numpy()))
            cons_gap_uq.append(np.abs(constraint_violation_uq.detach().cpu().numpy()))
            obj_gap.append(np.abs(self.get_obj_metric(data, vals, hard_non_negative=False).detach().cpu().numpy())) #TODO: look for changes (default False)

        obj_gap = np.concatenate(obj_gap, axis=0)
        cons_gap_eq = np.concatenate(cons_gap_eq, axis=0)
        cons_gap_uq = np.concatenate(cons_gap_uq, axis=0)
        return obj_gap, cons_gap_eq, cons_gap_uq