#!/usr/bin/env python3
# coding: utf-8
# @Author  : Yiming Li @ ICT, CAS
# @E-mail  : liyiming22s1@ict.ac.cn

import librosa
import os
from pathlib import Path
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import glob
import math
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
import sklearn.preprocessing as pre
from psds_eval import PSDSEval

def find_contiguous_regions(activity_array):
    change_indices = np.logical_xor(activity_array[1:],
                                    activity_array[:-1]).nonzero()[0]
    change_indices += 1

    if activity_array[0]:
        change_indices = np.r_[0, change_indices]

    if activity_array[-1]:
        change_indices = np.r_[change_indices, activity_array.size]

    return change_indices.reshape((-1, 2))


def binarize(x, threshold=0.5):
    if x.ndim == 3:
        return np.array(
            [pre.binarize(sub, threshold=threshold) for sub in x])
    else:
        return pre.binarize(x, threshold=threshold)


def median_filter(x, window_size, threshold=0.5):
    x = binarize(x, threshold=threshold)
    if x.ndim == 3: # (batch_size, time_steps, num_classes)
        size = (1, window_size, 1)
    elif x.ndim == 2 and x.shape[0] == 1: # (batch_size, time_steps)
        size = (1, window_size)
    elif x.ndim == 2 and x.shape[0] > 1: # (time_steps, num_classes)
        size = (window_size, 1)
    return scipy.ndimage.median_filter(x, size=size)


def predictions_to_time(df, ratio):
    if len(df) == 0:
        return df
    df.onset = df.onset * ratio
    df.offset = df.offset * ratio
    return df


def connect_clusters(x, n=1):
    if x.ndim == 1:
        return connect_clusters_(x, n)
    if x.ndim >= 2:
        return np.apply_along_axis(lambda a: connect_clusters_(a, n=n), -2, x)


def connect_clusters_(x, n=1):
    assert x.ndim == 1, "input needs to be 1d"
    reg = find_contiguous_regions(x)
    start_end = connect_(reg, n=n)
    zero_one_arr = np.zeros_like(x, dtype=int)
    for sl in start_end:
        zero_one_arr[sl[0]:sl[1]] = 1
    return zero_one_arr


def connect_(pairs, n=1):
    if len(pairs) == 0:
        return []
    start_, end_ = pairs[0]
    new_pairs = []
    for i, (next_item, cur_item) in enumerate(zip(pairs[1:], pairs[0:])):
        end_ = next_item[1]
        if next_item[0] - cur_item[1] <= n:
            pass
        else:
            new_pairs.append((start_, cur_item[1]))
            start_ = next_item[0]
    new_pairs.append((start_, end_))
    return new_pairs

def compute_psds(prediction_dfs,
                 ground_truth,
                 duration,
                 dtc_threshold=0.5,
                 gtc_threshold=0.5,
                 max_efpr=None,
                 save_dir=None):

    if not isinstance(ground_truth, pd.DataFrame):
        ground_truth = pd.read_csv(ground_truth, sep="\t")
    if not isinstance(duration, pd.DataFrame):
        duration = pd.read_csv(duration, sep="\t")

    aid_to_dur = dict(zip(duration["audio_id"], duration["duration"]))

    metadata = []
    for _, row in ground_truth.iterrows():
        dataid = row["filename"]
        aid = row["audio_id"]
        metadata.append({
            "filename": dataid,
            "duration": aid_to_dur[aid],
        })
    duration = pd.DataFrame(metadata)

    if "audio_id" in ground_truth:
        ground_truth = ground_truth.drop("audio_id", axis=1)

    psds_eval = PSDSEval(
        ground_truth=ground_truth,
        metadata=duration,
        dtc_threshold=dtc_threshold,
        gtc_threshold=gtc_threshold,
        cttc_threshold=0.0,
    )

    for i, k in enumerate(prediction_dfs.keys()):
        det = prediction_dfs[k]
        psds_eval.add_operating_point(
            det, info={"name": f"Op {i + 1:02d}", "threshold": k}
        )

    psds_score = psds_eval.psds(alpha_ct=0,
                                alpha_st=0,
                                max_efpr=max_efpr)

    return psds_score.value

class AudioDataset(Dataset):
    def __init__(self, root: str, download: bool = False):
        self.root = os.path.expanduser(root)

    def __getitem__(self, index):
        raise NotImplementedError

    def download(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

class Grouding_set(AudioDataset):
    base_folder = 'TAG'
    audio_dir = 'audio'
    eval_dir = 'test'
    json_file = {
        'filename': os.path.join('metadata','test.json')
    }
    def __init__(self,
                root,
                sample_rate: int = 32000):
        super().__init__(root)
        self.data = json.load(open(os.path.join(self.root, self.base_folder, self.eval_dir, self.json_file['filename'])))
        self.sample_rate = sample_rate
        self.generate_index()
        
    def generate_index(self):
        self.idxs = []
        for audio_idx, audio_item in enumerate(self.data):
            for phrase_idx, _ in enumerate(audio_item["phrases"]):
                self.idxs.append((audio_idx, phrase_idx))

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
            audio_time_series = audio_time_series[:start_index + audio_duration * sample_rate] 
        return torch.FloatTensor(audio_time_series)
    
    def __getitem__(self, index):
        audio_idx, phrase_idx = self.idxs[index]
        audio_item = self.data[audio_idx]
        audio_id = audio_item["audio_id"]
        audiocap_id = audio_item["audiocap_id"]
        audio_path = os.path.join(self.root, self.base_folder, self.eval_dir, self.audio_dir, audio_id)
        audio = self.load_audio_into_tensor(audio_path, resample=True)
        phrase_item = audio_item["phrases"][phrase_idx]
        phrase = phrase_item["phrase"]
        st_idx = phrase_item["start_index"]
        return audio_id, audio, phrase, str(audiocap_id), str(st_idx)

    def __len__(self):
        return len(self.idxs)

with open("settings/inference_grounding.yaml", "r") as f:
    config = yaml.safe_load(f)

device = config["device"]

model = ASE(config)
model.to(device)

cp_path = config["eval"]["ckpt"]
cp = torch.load(cp_path)
model.load_state_dict(cp['model'], strict=False)
model.eval()
print("Model weights loaded from {}".format(cp_path))

# Load dataset
eval_dataset = Grouding_set(root=config["eval"]["data_root_dir"])
eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=8)
ground_truth = []
for audio_item in eval_dataloader.dataset.data:
    audiocap_id = audio_item["audiocap_id"]
    audio_id = audio_item["audio_id"]
    for phrase_item in audio_item["phrases"]:
        start_index = phrase_item["start_index"]
        fname = f"{audiocap_id}_{start_index}"
        for onset, offset in phrase_item["segments"]:
            if onset == 0 and offset == 0:
                continue
            ground_truth.append({
                "filename": fname,
                "event_label": "fake_event",
                "onset": onset,
                "offset": offset,
                "audio_id": audio_id,
            })
ground_truth = pd.DataFrame(ground_truth)
audio_durations = config["eval"]["data_root_dir"] + "TAG/test/metadata/duration.csv"
n_thresholds = 50
thresholds = np.arange(
    1 / (n_thresholds * 2), 1, 1 / n_thresholds)
psds_buffer = {th: [] for th in thresholds}

window_size = 3
time_resolution = 10.0 / 32
n_connect = math.ceil(0.5 / time_resolution)

with torch.no_grad():
    model.eval()
    scores_raw_dic, scores_postprocessed_dic = {}, {}
    for audio_id, audio, caption, audiocap_id, start_index in tqdm(eval_dataloader, unit="batch", ascii=True, ncols=100, leave=False):
        audio = audio.to(device, non_blocking=True)
        _, frame_embeds = model.encode_audio(audio)
        frame_embeds = F.normalize(frame_embeds, dim=-1)
        _, word_embeds, attn_mask = model.encode_text(caption)
        text_embeds = model.msc(word_embeds, model.codebook, attn_mask)
        text_embeds = F.normalize(text_embeds, dim=-1)
        sim = text_embeds @ frame_embeds.squeeze(0).t()
        sim = F.sigmoid(sim)
        for th in thresholds:
            filtered_probs = median_filter(
                sim.cpu(),
                window_size=window_size,
                threshold=th
            )
            for idx in range(audio.size(0)):
                tmp_audiocap_id = audiocap_id[idx]
                tmp_start_index = start_index[idx]
                fname = f"{tmp_audiocap_id}_{tmp_start_index}"
                if fname not in ground_truth["filename"].unique():
                    continue
                change_indices = find_contiguous_regions(
                    connect_clusters(
                        filtered_probs[idx],
                        n_connect
                    )
                )                
                for row in change_indices:
                    psds_buffer[th].append({
                        "filename": fname,
                        "event_label": "fake_event",
                        "onset": row[0],
                        "offset": row[1]
                    })

for th in thresholds:
    if len(psds_buffer[th]) > 0:
        pred_df = pd.DataFrame(psds_buffer[th])
    else:
        pred_df = pd.DataFrame({
            "filename": [],
            "event_label": [],
            "onset": [],
            "offset": []
        })
    pred_df = predictions_to_time(pred_df, ratio=time_resolution)
    psds_buffer[th] = pred_df
       
psds1_student_sed_scores_eval = compute_psds(
    psds_buffer,
    ground_truth,
    audio_durations,
    dtc_threshold=0.7,
    gtc_threshold=0.7,
    max_efpr=800
)
psds2_student_sed_scores_eval = compute_psds(
    psds_buffer,
    ground_truth,
    audio_durations,
    dtc_threshold=0.1,
    gtc_threshold=0.1,
    max_efpr=800
)

psds_2021_student_sed_scores_eval = compute_psds(
    psds_buffer,
    ground_truth,
    audio_durations,
    dtc_threshold=0.5,
    gtc_threshold=0.5,
    max_efpr=800
)
print("psds1") 
print(psds1_student_sed_scores_eval)
print("psds2")
print(psds2_student_sed_scores_eval)
print("psds_2021")
print(psds_2021_student_sed_scores_eval)