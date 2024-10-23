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
import sed_scores_eval
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
from sed_scores_eval.utils.scores import create_score_dataframe

def compute_psds_from_scores(
    scores,
    ground_truth_file,
    durations_file,
    dtc_threshold=0.5,
    gtc_threshold=0.5,
    cttc_threshold=0.3,
    alpha_ct=0,
    alpha_st=0,
    max_efpr=100,
    num_jobs=4,
    save_dir=None,
):
    psds, psd_roc, single_class_rocs, *_ = sed_scores_eval.intersection_based.psds(
        scores=scores, ground_truth=ground_truth_file,
        audio_durations=durations_file,
        dtc_threshold=dtc_threshold, gtc_threshold=gtc_threshold,
        cttc_threshold=cttc_threshold, alpha_ct=alpha_ct, alpha_st=alpha_st,
        max_efpr=max_efpr, num_jobs=num_jobs,
    )
    return psds

def compute_collar_f1(    
    scores,
    ground_truth_file,
    collar = 0.2,
    offset_collar_rate = 0.2,
    time_decimals = 30):
    
    f_best, p_best, r_best, thresholds_best, stats_best = sed_scores_eval.collar_based.best_fscore(
        scores=scores,
        ground_truth=ground_truth_file,
        onset_collar=collar, offset_collar=collar,
        offset_collar_rate=offset_collar_rate,
        time_decimals=time_decimals,
        num_jobs=8,
    )
    return f_best

def compute_seg_f1(    
    scores,
    ground_truth_file,
    durations_file):
    
    f_best, p_best, r_best, thresholds_best, stats_best = sed_scores_eval.segment_based.best_fscore(
        scores=scores,
        ground_truth=ground_truth_file,
        audio_durations=durations_file,
        num_jobs=8,
    )
    return f_best
    
def post_process(c_scores):
    _, nf, nc = c_scores.size()
    c_scores = c_scores.squeeze(0).detach().cpu().numpy()
    scores_raw = create_score_dataframe(
            scores=c_scores,
            timestamps=[i * 10 / nf for i in range(nf + 1)],
            event_classes=eval_dataset.classes,
    )
        
    c_scores = scipy.ndimage.filters.median_filter(c_scores, (3, 1))
    scores_postprocessed = create_score_dataframe(
        scores=c_scores,
        timestamps=[i * 10 / nf for i in range(nf + 1)],
        event_classes=eval_dataset.classes,
    )
    return scores_raw, scores_postprocessed

class AudioDataset(Dataset):
    def __init__(self, root: str, download: bool = False):
        self.root = os.path.expanduser(root)

    def __getitem__(self, index):
        raise NotImplementedError

    def download(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
    
class AudioSet_SL_set(AudioDataset):
    base_folder = 'AudioSet_SL'
    audio_dir = 'audio'
    eval_dir = 'AudioSet_SL_eval'
    label_col = 'event_label'
    file_col = 'filename'
    meta = {
        'filename': os.path.join('metadata','audioset_eval_strong.tsv')
    }
    meta_dur = {
        'filename': os.path.join('metadata','audioset_eval_strong_durations.tsv')
    }

    def __init__(self, root, reading_transformations: nn.Module = None, download: bool = True):
        super().__init__(root)
        self._load_meta()

        self.targets, self.audio_paths = [], []
        print("Loading audio files")
        for _, row in tqdm(self.df.iterrows()):
            file_path = os.path.join(self.root, self.base_folder, self.audio_dir, self.eval_dir, row[self.file_col])
            self.audio_paths.append(file_path)
            
        self.audio_paths = list(set(self.audio_paths))

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        self.df = pd.read_csv(path, sep='\t')
        self.class_to_idx = {}
        self.classes = [x for x in sorted([i for i in self.df[self.label_col].unique() if type(i) == str])]
        for i, category in enumerate(self.classes):
            self.class_to_idx[category] = i
            
    def load_audio_into_tensor(self, audio_path, audio_duration=10, resample=False):
        audio_time_series, sample_rate = torchaudio.load(audio_path)
        resample_rate = 32000
        
        if resample:
            resampler = T.Resample(sample_rate, resample_rate)
            audio_time_series = resampler(audio_time_series)
            sample_rate = resample_rate
            
        audio_time_series = audio_time_series.mean(dim=0).reshape(-1)
        if audio_duration * sample_rate >= audio_time_series.shape[0]:
            repeat_factor = int(np.ceil((audio_duration*sample_rate) / audio_time_series.shape[0]))
            audio_time_series = audio_time_series.repeat(repeat_factor)
            audio_time_series = audio_time_series[0 : audio_duration * sample_rate]
            
        else:
            audio_time_series = audio_time_series[:audio_duration * sample_rate] 
        return torch.FloatTensor(audio_time_series)
    
    def __getitem__(self, index):
        file_path = self.audio_paths[index]
        audio = self.load_audio_into_tensor(file_path, resample=True)
        return audio, file_path.split("/")[-1]

    def __len__(self):
        return len(self.audio_paths)
   
    
with open("settings/inference_sed.yaml", "r") as f:
    config = yaml.safe_load(f)

device = config["device"]
model = ASE(config)
model.to(device)

cp_path = config["eval"]["ckpt"]
cp = torch.load(cp_path)
model.load_state_dict(cp['model'], strict=True)
model.eval()
print("Model weights loaded from {}".format(cp_path))

# Load dataset
eval_dataset = AudioSet_SL_set(root=config["eval"]["data_root_dir"], download=False)
    
classes = [x.replace('_',' ').lower() for x in eval_dataset.classes]
eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=8)

ground_truth = sed_scores_eval.io.read_ground_truth_events(config["eval"]["data_root_dir"] + "AudioSet_SL/metadata/audioset_eval_strong.tsv")
audio_durations = sed_scores_eval.io.read_audio_durations(config["eval"]["data_root_dir"] + "AudioSet_SL/metadata/audioset_eval_strong_durations.tsv")   

ground_truth = {
    audio_id: gt for audio_id, gt in ground_truth.items()
    if len(gt) > 0
}
audio_durations = {
    audio_id: audio_durations[audio_id]
    for audio_id in ground_truth.keys()
}

with torch.no_grad():
    model.eval()
    _, word_embeds, attn_mask = model.encode_text(classes)
    text_embeds = model.msc(word_embeds, model.codebook, attn_mask)
    text_embeds = F.normalize(text_embeds, dim=-1)
    scores_raw_dic, scores_postprocessed_dic = {}, {}
    for i, (audio, filename) in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
        audio = audio.to(device, non_blocking=True)
        _, frame_embeds = model.encode_audio(audio)
        audio_embeds = model.msc(frame_embeds.unsqueeze(1), model.codebook)
        frame_embeds = F.normalize(frame_embeds, dim=-1)
        similarity = frame_embeds @ text_embeds.t()
        scores_raw, scores_postprocessed = post_process(similarity)
        scores_raw_dic[filename[0].split(".flac")[0]] = scores_raw
        scores_postprocessed_dic[filename[0].split(".flac")[0]] = scores_postprocessed
    pop_lst = []

    for k in scores_postprocessed_dic.keys():
        if k not in ground_truth.keys():
            pop_lst.append(k)
    for k in pop_lst:       
            scores_postprocessed_dic.pop(k)

print("psds1")        
psds1_student_sed_scores_eval = compute_psds_from_scores(
    scores_postprocessed_dic,
    ground_truth,
    audio_durations,
    dtc_threshold=0.7,
    gtc_threshold=0.7,
    cttc_threshold=None,
    alpha_ct=0,
    alpha_st=1,
)
print(psds1_student_sed_scores_eval)

print("psds2")
psds2_student_sed_scores_eval = compute_psds_from_scores(
    scores_postprocessed_dic,
    ground_truth,
    audio_durations,
    dtc_threshold=0.1,
    gtc_threshold=0.1,
    cttc_threshold=0.3,
    alpha_ct=0.5,
    alpha_st=1,
)
print(psds2_student_sed_scores_eval)

print("event")
eb_f1 = compute_collar_f1(scores_postprocessed_dic, ground_truth)
print(eb_f1)

print("seg")
seg_f1 = compute_seg_f1(scores_postprocessed_dic, ground_truth, audio_durations)
print(seg_f1)