#!/usr/bin/env python3
# coding: utf-8
# @Author  : Yiming Li @ ICT, CAS
# @E-mail  : liyiming22s1@ict.ac.cn

import librosa
import os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import glob
import random
import torch
from ruamel import yaml
from tqdm import tqdm
import json
from re import sub
from models.ase_model import ASE
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
    
with open("settings/inference_example.yaml", "r") as f:
    config = yaml.safe_load(f)

device = config["device"]
model = ASE(config)
model.to(device)

cp_path = config["eval"]["ckpt"]
cp = torch.load(cp_path)
model.load_state_dict(cp['model'], strict=True)
model.eval()
print("Model weights loaded from {}".format(cp_path))
    
classes = ['male speech', 'female speech', 'electric shaver']
audio_path = 'example_audio/example.wav'
audio_time_series, sample_rate = torchaudio.load(audio_path)
resampler = T.Resample(sample_rate, 32000)
audio_time_series = resampler(audio_time_series)
audio_time_series = audio_time_series.mean(dim=0).reshape(-1)
audio_time_series = audio_time_series.to(device, non_blocking=True)
model.eval()
_, word_embeds, attn_mask = model.encode_text(classes)
text_embeds = model.msc(word_embeds, model.codebook, attn_mask)
text_embeds = F.normalize(text_embeds, dim=-1)
_, frame_embeds = model.encode_audio(audio_time_series.unsqueeze(0))
audio_embeds = model.msc(frame_embeds, model.codebook)
frame_embeds = F.normalize(frame_embeds, dim=-1)
audio_embeds = F.normalize(audio_embeds, dim=-1)
frame_similarity = frame_embeds @ text_embeds.t()
clip_similarity = audio_embeds @ text_embeds.t()
print(frame_similarity, clip_similarity)