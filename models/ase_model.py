#!/usr/bin/env python3
# coding: utf-8
# @Author  : Yiming Li @ ICT, CAS
# @E-mail  : liyiming22s1@ict.ac.cn

import math
import torch
import torch.nn as nn
from models.audio_encoder import AudioEncoder
from models.text_encoder import TextEncoder
import torch.nn.functional as F
import copy
from tools.losses import AudioTextContrastiveLoss_HN
from tools.utils import *
from torch import distributed as dist

class Sparsemax(nn.Module):
    def __init__(self, dim=None):
        super(Sparsemax, self).__init__()
        self.dim = -1 if dim is None else dim

    def forward(self, input, device='cuda'):
        input = input.transpose(0, self.dim)
        original_size = input.size()
        input = input.reshape(input.size(0), -1)
        input = input.transpose(0, 1)
        dim = 1

        number_of_logits = input.size(dim)

        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(start=1, end=number_of_logits + 1, step=1, device=device, dtype=input.dtype).view(1, -1)
        range = range.expand_as(zs)

        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        zs_sparse = is_gt * zs

        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        self.output = torch.max(torch.zeros_like(input), input - taus)

        output = self.output
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)

        return output

    def backward(self, grad_output):
        dim = 1
        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input
    
class Modality_Shared_Codebook(nn.Module):
    def __init__(self, code_dim, T=1000.0):
        super().__init__()
        self.attn_activation = Sparsemax(dim=-1)
        self.attn_dim = code_dim
        self.T = T

    def forward(self, feats, codebook, mask=None):
        q = feats
        k = codebook 
        k = k.unsqueeze(0) 
        k = k.transpose(2, 1) 
        
        # equation 2 in our paper
        attn = torch.matmul(q, k) 

        attn = attn / math.sqrt(self.attn_dim) 
        if mask is not None:    
            attn = attn * mask.unsqueeze(-1) 
        
        attn = attn / self.T 
        
        # maxpooling
        attn = attn.max(1)[0]

        # sparse constraints, equation 3 in our paper
        attn_weight = self.attn_activation(attn)

        # aggregation
        attn_feats = attn_weight @ codebook  

        return attn_feats

class ASE(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.audio_encoder = AudioEncoder(config)
        self.text_encoder = TextEncoder(config)

        embed_size = config["embed_size"]
        proj_size = embed_size
        audio_width = self.audio_encoder.audio_width
        text_width = self.text_encoder.text_width

        self.frame_proj = nn.Sequential(
            nn.LayerNorm(audio_width),
            nn.Linear(audio_width, embed_size),
            nn.GELU(),
            nn.LayerNorm(embed_size),
            nn.Linear(embed_size, proj_size),
        )

        self.word_proj = nn.Sequential(
            nn.LayerNorm(text_width),
            nn.Linear(text_width, embed_size),
            nn.GELU(),
            nn.LayerNorm(embed_size),
            nn.Linear(embed_size, proj_size),
        )

        self.temp = nn.Parameter(torch.ones([]) * config["temp"])

        self.code_num = 4096
        self.codebook = nn.Parameter(torch.randn(self.code_num, proj_size))
        self.msc = Modality_Shared_Codebook(code_dim=proj_size)
        
        self.embed_reg = config["embed_regularization"]
        
        # use the hard-negative guided loss
        self.atc_loss = AudioTextContrastiveLoss_HN() 

    def encode_audio(self, audio):
        audio_feats, frame_feats = self.audio_encoder(audio)
        frame_embeds = self.frame_proj(frame_feats)
        return audio_feats, frame_embeds

    def encode_text(self, text):
        text_feats, attn_mask = self.text_encoder(text)
        word_embeds = self.word_proj(text_feats[:, 1:, :])
        return text_feats, word_embeds, attn_mask
       
    def forward(self, audio, text, idx):
        _, frame_embeds = self.encode_audio(audio)
        _, word_embeds, attn_mask = self.encode_text(text)

        idx = idx.view(-1, 1)
        audio_embeds = self.msc(frame_embeds, self.codebook)
        text_embeds = self.msc(word_embeds, self.codebook, attn_mask)
        audio_embeds = F.normalize(audio_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        pos_idx = torch.eq(idx, idx.t()).float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

        global_sim_a2t = audio_embeds @ text_embeds.t() / self.temp
        global_sim_t2a = text_embeds @ audio_embeds.t() / self.temp
        
        loss = self.atc_loss(global_sim_a2t, global_sim_t2a, sim_targets)
        if self.embed_reg:
            loss = loss + torch.mean(torch.abs(audio_embeds)) / torch.sqrt(torch.sum(audio_embeds**2)) + \
                torch.mean(torch.abs(text_embeds)) / torch.sqrt(torch.sum(text_embeds**2))
        return loss
