#!/usr/bin/env python3
# coding: utf-8
# @Author  : Yiming Li @ ICT, CAS
# @E-mail  : liyiming22s1@ict.ac.cn

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import util


class AudioTextContrastiveLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self,
                sim_a2t,
                sim_t2a,
                sim_targets=None,
                label_smooth=False):
        if sim_targets is None:
            sim_targets = torch.zeros(sim_a2t.size()).to(
                sim_a2t.device
            )
            sim_targets.fill_diagonal_(1)

        loss_a2t = - torch.sum(
            F.log_softmax(sim_a2t, dim=1) * sim_targets, dim=1
        ).mean()

        loss_t2a = - torch.sum(
            F.log_softmax(sim_t2a, dim=1) * sim_targets, dim=1
        ).mean()

        loss_atc = (loss_a2t + loss_t2a) / 2
        
        if label_smooth:
            loss_a2t_ls = - torch.mean(
                F.log_softmax(sim_a2t, dim=1), dim=1
            ).mean()
            
            loss_t2a_ls = - torch.mean(
                F.log_softmax(sim_t2a, dim=1), dim=1
            ).mean()
            
            loss_atc = 0.9 * loss_atc + 0.1 * (loss_a2t_ls + loss_t2a_ls) / 2
            
        return loss_atc

class AudioTextContrastiveLoss_HN(nn.Module):

    def __init__(self):
        super().__init__()

    def reweight(self, sim_mat, sim_targets, beta=0.15):
        beta_sim_mat = sim_mat * beta
        exp_beta_sim_mat = torch.exp(beta_sim_mat)
        bs = sim_mat.size(0)
        exp_beta_sim_mat_1 = (bs - 1) * exp_beta_sim_mat
        exp_beta_sim_mat_2 = torch.sum(exp_beta_sim_mat, 1) - exp_beta_sim_mat
        weights = exp_beta_sim_mat_1 / exp_beta_sim_mat_2
        weights[sim_targets > 0] = 1.0
        return weights


    def forward(self,
                sim_a2t,
                sim_t2a,
                sim_targets=None):
        if sim_targets is None:
            sim_targets = torch.zeros(sim_a2t.size()).to(
                sim_a2t.device
            )
            sim_targets.fill_diagonal_(1)

        logits_max_a2t, _ = torch.max(sim_a2t, dim=1, keepdim=True)
        sim_a2t = sim_a2t - logits_max_a2t.detach()
        weights_a2t = self.reweight(sim_a2t, sim_targets)

        logits_max_t2a, _ = torch.max(sim_t2a, dim=1, keepdim=True)
        sim_t2a = sim_t2a - logits_max_t2a.detach()
        weights_t2a = self.reweight(sim_t2a, sim_targets)

        exp_logits_a2t = torch.exp(sim_a2t)
        log_prob_a2t = sim_a2t - torch.log(1e-7 + (weights_a2t * exp_logits_a2t).sum(1, keepdim=True))

        exp_logits_t2a = torch.exp(sim_t2a)
        log_prob_t2a = sim_t2a - torch.log(1e-7 + (weights_t2a * exp_logits_t2a).sum(1, keepdim=True))

        loss_a2t = -(sim_targets * log_prob_a2t).sum(1).mean()
        loss_t2a = -(sim_targets * log_prob_t2a).sum(1).mean()

        loss_atc = (loss_a2t + loss_t2a) / 2
            
        return loss_atc
    
    
class NTXent(nn.Module):

    def __init__(self, temperature=0.07):
        super(NTXent, self).__init__()
        self.loss = nn.LogSoftmax(dim=1)
        self.tau = temperature

    def forward(self, audio_embeds, text_embeds, labels):

        n = audio_embeds.shape[0]

        a2t = util.cos_sim(audio_embeds, text_embeds) / self.tau
        t2a = util.cos_sim(text_embeds, audio_embeds) / self.tau

        t2a_loss = - self.loss(t2a).mean()
        a2t_loss = - self.loss(a2t).mean()


        loss = 0.5 * a2t_loss + 0.5 * t2a_loss

        return loss
