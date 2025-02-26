#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import json
import torch
import random
import librosa
from torch.utils.data import Dataset
from data_handling.text_transform import text_preprocess
import torchaudio.transforms as T
import torchaudio

class AudioCaptionDataset(Dataset):

    def __init__(self, dataset, split):
        super().__init__()
        self.dataset = dataset
        self.split = split
        self.sr = 32000

        json_path = f"downstream_data/{dataset}/json_files/{split}.json"
        self.max_length = 10 * self.sr


        with open(json_path, 'r') as f:
            json_obj = json.load(f)
            self.num_captions_per_audio = json_obj["num_captions_per_audio"]
            json_obj = json_obj["data"]

        if self.num_captions_per_audio == 1:
            self.captions = [item["caption"] for item in json_obj]
            self.wav_paths = [item["audio"] for item in json_obj]
        elif self.num_captions_per_audio == 5:
            self.captions = [item["caption_{}".format(i)] for item in json_obj for i in range(1, 6)]
            self.wav_paths = [item["audio"] for item in json_obj for _ in range(1, 6)]
        else:
            raise ValueError("Incorrect num_captions_per_audio.")

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        audio_id = index // self.num_captions_per_audio
        audio_name = self.wav_paths[index].split("/")[-1]
        wav_path = self.wav_paths[index]

        waveform, sr = torchaudio.load(wav_path)
        # print(wav_path, waveform.size())
        waveform = waveform.mean(dim=tuple(range(waveform.ndim-1))).reshape(-1)
        resampler = T.Resample(sr, self.sr)
        if waveform.size(0) != 0:
            waveform = resampler(waveform)
            
        if self.max_length != 0:
            # if audio length is longer than max_length, we randomly crop it to mac length
            if waveform.shape[-1] > self.max_length:
                max_start = waveform.shape[-1] - self.max_length
                start = random.randint(0, max_start)
                waveform = waveform[start: start + self.max_length]

        caption = text_preprocess(self.captions[index])

        return waveform, caption, audio_id

