import numpy as np
from tqdm import tqdm
from glob import glob
import os.path as osp
import audio
from hparams import hparams

import warnings
warnings.filterwarnings("ignore")

audio_dir = "/d/dataset/audio/CMLRdataset/audios/x/aac"
audio_files = glob(osp.join(audio_dir, "*"))
for af in tqdm(audio_files, total=len(audio_files)):
    wav = audio.load_wav(af, hparams.sample_rate)
    mel = audio.melspectrogram(wav).T
    np.save(af.replace(".aac", ".npy").replace("aac", "npy"), mel)
