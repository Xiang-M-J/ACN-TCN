"""
Torch dataset object for synthetically rendered spatial data.
"""

import os
import json
import random
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scaper
import torch
import torchaudio
import torchaudio.transforms as AT
from random import randrange
import argparse


import tqdm
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
dsets = ['train', 'val', 'test']
input_dir = "data/FSDSoundScapes"
resampler = AT.Resample(44100, 16000)

_labels = [
    "Acoustic_guitar", "Applause", "Bark", "Bass_drum",
    "Burping_or_eructation", "Bus", "Cello", "Chime", "Clarinet",
    "Computer_keyboard", "Cough", "Cowbell", "Double_bass",
    "Drawer_open_or_close", "Electric_piano", "Fart", "Finger_snapping",
    "Fireworks", "Flute", "Glockenspiel", "Gong", "Gunshot_or_gunfire",
    "Harmonica", "Hi-hat", "Keys_jangling", "Knock", "Laughter", "Meow",
    "Microwave_oven", "Oboe", "Saxophone", "Scissors", "Shatter",
    "Snare_drum", "Squeak", "Tambourine", "Tearing", "Telephone",
    "Trumpet", "Violin_or_fiddle", "Writing"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("TAU_path", default="E:/xmj/datasets/TAU")
    args = parser.parse_args()

    for dset in dsets:
        print("processing {}".format(dset))
        fg_dir = f"data/FSDSoundScapes/FSDKaggle2018/{dset}"
        if dset in ['train', 'val']:
            bg_dir = os.path.join(args.TAU_path, "TAU-urban-acoustic-scenes-2019-development") 
        else:
            bg_dir = os.path.join(args.TAU_path, "TAU-urban-acoustic-scenes-2019-evaluation")

        samples = sorted(list(Path(os.path.join(input_dir, 'jams', dset)).glob('[0-9]*')))
        for sample_path in tqdm.tqdm(samples):
            jamsfile = os.path.join(sample_path, 'mixture.jams')

            mixture, jams, ann_list, event_audio_list = scaper.generate_from_jams(
                jamsfile, fg_path=fg_dir, bg_path=bg_dir, disable_sox_warnings=True)

            sources = [[], []]

            sources[0].append(resampler(torch.from_numpy(event_audio_list[0]).float().T).numpy())
            sources[1].append(-1)
            for i in range(len(ann_list)):
                sources[0].append(resampler(torch.from_numpy(event_audio_list[i+1]).float().T).numpy())
                sources[1].append(_labels.index(ann_list[i][2]))
            npyfile = os.path.join(sample_path, 'mixture.npy')
            labelfile = os.path.join(sample_path, 'label.npy')
            np.save(npyfile, np.array(sources[0]))
            np.save(labelfile, np.array(sources[1]))
