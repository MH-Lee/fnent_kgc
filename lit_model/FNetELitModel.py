from logging import debug
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
from collections import defaultdict as ddict
from neuralkg.lit_model.BaseLitModel import BaseLitModel
from neuralkg.eval_task import *
from IPython import embed
from .utils import CosineAnnealingWarmUpRestarts

from functools import partial

class FNetELitModel(BaseLitModel):
    def __init__(self, model, args):
        super().__init__(model, args)
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        sample = batch["sample"]
        label  = batch["label"]
        sample_score = self.model(sample)
        label = ((1.0 - self.args.smoothing) * label) + (
                1.0 / self.args.num_ent
        )
        loss = self.loss(sample_score,label)
        self.log("Train|loss", loss,  on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # pos_triple, tail_label, head_label = batch
        results = dict()
        ranks = link_predict(batch, self.model, prediction='tail')
        results["count"] = torch.numel(ranks)
        results["mrr"] = torch.sum(1.0 / ranks).item()
        for k in self.args.calc_hits:
            results['hits@{}'.format(k)] = torch.numel(ranks[ranks <= k])
        return results
    
    def validation_epoch_end(self, results) -> None:
        outputs = self.get_results(results, "Eval")
        # self.log("Eval|mrr", outputs["Eval|mrr"], on_epoch=True)
        self.log_dict(outputs, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        results = dict()
        ranks = link_predict(batch, self.model, prediction='tail')
        results["count"] = torch.numel(ranks)
        results["mrr"] = torch.sum(1.0 / ranks).item()
        results["mr"] = torch.sum(ranks).item()
        for k in self.args.calc_hits:
            results['hits@{}'.format(k)] = torch.numel(ranks[ranks <= k])
        return results
    
    def test_epoch_end(self, results) -> None:
        outputs = self.get_results(results, "Test")
        self.log_dict(outputs, prog_bar=True, on_epoch=True)
    
    '''lr_scheduler'''
    def configure_optimizers(self):
        milestones = int(self.args.max_epochs // 10)
        optimizer = self.optimizer_class(self.model.parameters(), lr=self.args.lr, weight_decay=0.0)
        CosineAnnealingLR = CosineAnnealingWarmUpRestarts(optimizer, T_0=milestones, T_mult=2, eta_max=self.args.lr, T_up=5, gamma=0.8)
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': CosineAnnealingLR}
        return optim_dict

    # def configure_optimizers(self):
    #     milestones = int(self.args.max_epochs / 2)
    #     optimizer = self.optimizer_class(self.model.parameters(), lr=self.args.lr, weight_decay=0)
    #     StepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[milestones], gamma=0.5)
    #     optim_dict = {'optimizer': optimizer, 'lr_scheduler': StepLR}
    #     return optim_dict