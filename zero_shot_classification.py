#!/usr/bin/env python3
# coding: utf-8
# @Author  : Yiming Li @ ICT, CAS
# @E-mail  : liyiming22s1@ict.ac.cn

import librosa
import numpy as np
import pandas as pd
import glob
import torch
import torchaudio
from ruamel import yaml
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import json
from re import sub
from data_handling.text_transform import text_preprocess
from models.ase_model import ASE
import torch.nn.functional as F
import torchaudio.transforms as T

with open("settings/inference_cls.yaml", "r") as f:
    config = yaml.safe_load(f)

device = config["device"]
model = ASE(config)
model.to(device)

cp_path = config["eval"]["ckpt"]
cp = torch.load(cp_path)
model.load_state_dict(cp['model'], strict=True)
model.eval()
print("Model weights loaded from {}".format(cp_path))

# ESC-50
df = pd.read_csv(config["eval"]["data_root_dir"] + 'ESC-50-master/meta/esc50.csv')
class_to_idx = {}
sorted_df = df.sort_values(by=['target'])
classes = [x.replace('_', ' ') for x in sorted_df['category'].unique()]
for i, category in enumerate(classes):
    class_to_idx[category] = i
classes = [c + " can be heard" for c in classes] 

pre_path = config["eval"]["data_root_dir"] + 'ESC-50-master/audio/'
with torch.no_grad():
    _, word_embeds, attn_mask = model.encode_text(classes)
    text_embeds = model.msc(word_embeds, model.codebook, attn_mask)
    text_embeds = F.normalize(text_embeds, dim=-1)
    fold_acc = []
    for fold in range(1, 6):
        fold_df = sorted_df[sorted_df['fold'] == fold]
        y_preds, y_labels = [], []
        for file_path, target in tqdm(zip(fold_df["filename"], fold_df["target"]), total=len(fold_df)):
            audio_path = pre_path + file_path
            one_hot_target = torch.zeros(len(classes)).scatter_(0, torch.tensor(target), 1).reshape(1, -1)
            audio, sr = torchaudio.load(audio_path)
            audio = audio.mean(dim=0).reshape(-1)
            resampler = T.Resample(sr, 32000)
            audio = resampler(audio)
            audio = audio.to(device)
            if audio.shape[-1] < 32000 * 10:
                pad_length = 32000 * 10 - audio.shape[-1]
                audio = F.pad(audio, [0, pad_length], "constant", 0.0)
            audio = audio.reshape(1, -1)
            _, frame_embeds = model.encode_audio(audio)
            audio_embeds = model.msc(frame_embeds, model.codebook)
            audio_embeds = F.normalize(audio_embeds, dim=-1)
            similarity = audio_embeds @ text_embeds.t()
            y_pred = F.softmax(similarity.detach().cpu(), dim=1).numpy()
            y_preds.append(y_pred)
            y_labels.append(one_hot_target.cpu().numpy())

        y_labels, y_preds = np.concatenate(y_labels, axis=0), np.concatenate(y_preds, axis=0)
        acc = accuracy_score(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))
        print('Fold {} Accuracy {}'.format(fold, acc))
        fold_acc.append(acc)

print('ESC-50 Accuracy {}'.format(np.mean(np.array(fold_acc))))

# UrbanSound 8K
df = pd.read_csv(config["eval"]["data_root_dir"] + 'UrbanSound8K/metadata/UrbanSound8K.csv')
class_to_idx = {}
sorted_df = df.sort_values(by=['classID'])
classes = [x.replace('_', ' ') for x in sorted_df['class'].unique()]
for i, category in enumerate(classes):
    class_to_idx[category] = i
classes = [c for c in classes]

pre_path = config["eval"]["data_root_dir"] + 'UrbanSound8K/audio/'
with torch.no_grad():
    _, word_embeds, attn_mask = model.encode_text(classes)
    text_embeds = model.msc(word_embeds, model.codebook, attn_mask)
    text_embeds = F.normalize(text_embeds, dim=-1)
    fold_acc = []
    for fold in range(1, 11):
        fold_df = sorted_df[sorted_df['fold'] == fold]
        y_preds, y_labels = [], []
        for file_path, target in tqdm(zip(fold_df["slice_file_name"], fold_df["classID"]), total=len(fold_df)):
            audio_path = pre_path + "fold" + str(fold) + "/" + file_path
            one_hot_target = torch.zeros(len(classes)).scatter_(0, torch.tensor(target), 1).reshape(1, -1)
            audio, sr = torchaudio.load(audio_path)
            audio = audio.mean(dim=0).reshape(-1)
            resampler = T.Resample(sr, 32000)
            audio = resampler(audio)
            audio = audio.to(device)
            if audio.shape[-1] < 32000 * 10:
                pad_length = 32000 * 10 - audio.shape[-1]
                audio = F.pad(audio, [0, pad_length], "constant", 0.0)
            audio = audio.reshape(1, -1)
            _, frame_embeds = model.encode_audio(audio)
            audio_embeds = model.msc(frame_embeds, model.codebook)
            audio_embeds = F.normalize(audio_embeds, dim=-1)
            similarity = audio_embeds @ text_embeds.t()
            y_pred = F.softmax(similarity.detach().cpu(), dim=1).numpy()
            y_preds.append(y_pred)
            y_labels.append(one_hot_target.cpu().numpy())

        y_labels, y_preds = np.concatenate(y_labels, axis=0), np.concatenate(y_preds, axis=0)
        acc = accuracy_score(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))
        print('Fold {} Accuracy {}'.format(fold, acc))
        fold_acc.append(acc)

print('Urbansound8K Accuracy {}'.format(np.mean(np.array(fold_acc))))

# VggSound
df = pd.read_csv(config["eval"]["data_root_dir"] + 'VGGSound/metadata/vggsound_test.csv', names=['filename', 'class'])
class_to_idx = {}
sorted_df = df.sort_values(by=['class'])
classes = [x.replace('_', ' ') for x in sorted_df['class'].unique()]
for i, category in enumerate(classes):
    class_to_idx[category] = i
classes = [c + " can be heard" for c in classes]
df['target'] = 0
for idx, row in df.iterrows():
    df.at[idx, 'target'] = class_to_idx[df.at[idx, 'class']]

pre_path = config["eval"]["data_root_dir"] + 'VGGSound/audio/'
with torch.no_grad():
    _, word_embeds, attn_mask = model.encode_text(classes)
    text_embeds = model.msc(word_embeds, model.codebook, attn_mask)
    text_embeds = F.normalize(text_embeds, dim=-1)
    y_preds, y_labels = [], []
    for file_path, target in tqdm(zip(df["filename"], df["target"]), total=len(df)):
        audio_path = pre_path + file_path[:-4] + ".wav"
        one_hot_target = torch.zeros(len(classes)).scatter_(0, torch.tensor(target), 1).reshape(1, -1)
        audio, sr = torchaudio.load(audio_path)
        audio = audio.mean(dim=0).reshape(-1)
        resampler = T.Resample(sr, 32000)
        audio = resampler(audio)
        audio = audio.to(device)
        if audio.shape[-1] < 32000 * 10:
            pad_length = 32000 * 10 - audio.shape[-1]
            audio = F.pad(audio, [0, pad_length], "constant", 0.0)
        audio = audio.reshape(1, -1)
        audio = audio[:, :32000 * 10]
        _, frame_embeds = model.encode_audio(audio)
        audio_embeds = model.msc(frame_embeds, model.codebook)
        audio_embeds = F.normalize(audio_embeds, dim=-1)
        similarity = audio_embeds @ text_embeds.t()
        y_pred = F.softmax(similarity.detach().cpu(), dim=1).numpy()
        y_preds.append(y_pred)
        y_labels.append(one_hot_target.cpu().numpy())
        
    y_labels, y_preds = np.concatenate(y_labels, axis=0), np.concatenate(y_preds, axis=0)
    acc = accuracy_score(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))

print('VGGSound Accuracy {}'.format(acc))
