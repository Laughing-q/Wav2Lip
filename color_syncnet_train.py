import os.path as osp
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset

from models import SyncNet_color as SyncNet
import audio

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np

from glob import glob

import os
import random
import cv2
from hparams import hparams

import warnings

warnings.filterwarnings("ignore")


global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
print("use_cuda: {}".format(use_cuda))

window_size = 5
syncnet_mel_step_size = 16


class SyncDataset(Dataset):
    """SyncDataset

    Attributes:
        root
        └── images
            └── id
                └── *.jpg
        └── audios
            └── id.wav/aac
    """

    def __init__(self, audio_dir):
        self.audio_files = glob(osp.join(audio_dir, "*"))

    def get_frame_id(self, im_file):
        return int(Path(im_file).with_suffix("").name)

    def crop_audio_window(self, spec, im_file):
        start_frame_num = self.get_frame_id(im_file)
        start_idx = int(80.0 * (start_frame_num / float(hparams.fps)))
        end_idx = start_idx + syncnet_mel_step_size
        return spec[start_idx:end_idx, :]

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        im_dir = str(Path(audio_file.replace("audios", "images")).with_suffix(""))

        im_files = glob(osp.join(im_dir, "*"))
        if len(im_files) <= 3 * window_size:
            return None, None, None
        pos_idx, neg_idx = random.choices(range(len(im_files))[:-window_size], k=2)
        while neg_idx == pos_idx:
            pos_idx, neg_idx = random.choices(im_files, k=2)

        if random.random() < 0.5:
            y = torch.ones(1, dtype=torch.float32)
            chosen = pos_idx
        else:
            y = torch.zeros(1, dtype=torch.float32)
            chosen = neg_idx

        window_files = im_files[chosen : chosen + window_size]

        window = []
        for fname in window_files:
            im = cv2.imread(fname)
            im = cv2.resize(im, (hparams.img_size, hparams.img_size))
            window.append(im)

        wav = audio.load_wav(audio_file, hparams.sample_rate)

        orig_mel = audio.melspectrogram(wav).T

        mel = self.crop_audio_window(orig_mel.copy(), im_files[pos_idx])

        # H x W x 3 * T
        x = np.concatenate(window, axis=2)
        x = x.transpose(2, 0, 1)  # 3 * T, H, W
        x = x[:, x.shape[1] // 2 :]  # 3 * T, H // 2, W

        x = torch.from_numpy(x)
        mel = torch.from_numpy(mel.T).unsqueeze(0)

        return x, mel, y


logloss = nn.BCELoss()


def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss


def train(
    device,
    model,
    train_data_loader,
    test_data_loader,
    optimizer,
    checkpoint_dir=None,
    checkpoint_interval=None,
    nepochs=None,
):
    global global_step, global_epoch
    resumed_step = global_step

    while global_epoch < nepochs:
        running_loss = 0.0
        prog_bar = tqdm(enumerate(train_data_loader))
        for step, (x, mel, y) in prog_bar:
            if x is None:
                continue
            model.train()
            optimizer.zero_grad()

            # Transform data to CUDA device
            x = x.to(device)

            mel = mel.to(device)

            a, v = model(mel, x)
            y = y.to(device)

            loss = cosine_loss(a, v, y)
            loss.backward()
            optimizer.step()

            global_step += 1
            running_loss += loss.item()

            if global_step == 1 or global_step % checkpoint_interval == 0:
                save_checkpoint(model, optimizer, global_step, checkpoint_dir, global_epoch)

            # if global_step % hparams.syncnet_eval_interval == 0:
            #     with torch.no_grad():
            #         eval_model(test_data_loader, global_step, device, model, checkpoint_dir)

            prog_bar.set_description("Loss: {}".format(running_loss / (step + 1)))

        global_epoch += 1


def eval_model(test_data_loader, global_step, device, model, checkpoint_dir):
    eval_steps = 1400
    print("Evaluating for {} steps".format(eval_steps))
    losses = []
    while 1:
        for step, (x, mel, y) in enumerate(test_data_loader):
            model.eval()

            # Transform data to CUDA device
            x = x.to(device)

            mel = mel.to(device)

            a, v = model(mel, x)
            y = y.to(device)

            loss = cosine_loss(a, v, y)
            losses.append(loss.item())

            if step > eval_steps:
                break

        averaged_loss = sum(losses) / len(losses)
        print(averaged_loss)

        return


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):
    checkpoint_path = osp.join(checkpoint_dir, "checkpoint_step{:09d}.pth".format(global_step))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer": optimizer_state,
            "global_step": step,
            "global_epoch": epoch,
        },
        checkpoint_path,
    )
    print("Saved checkpoint:", checkpoint_path)


def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint


def load_checkpoint(path, model, optimizer, reset_optimizer=False):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]

    return model


if __name__ == "__main__":
    checkpoint_dir = "runs/"
    checkpoint_path = None

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    device = torch.device("cuda" if use_cuda else "cpu")
    # Model
    model = SyncNet().to(device)
    model.eval()
    print(
        "total trainable params {}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )
    # Dataset and Dataloader setup
    train_dataset = SyncDataset("/d/dataset/audio/HDTF_DATA/RD25_audios")
    for i, (im, mel, y) in enumerate(train_dataset):
        # print(i, x.shape, mel.shape, y)
        im = im.to(device).float() / 255.
        mel = mel.to(device).float()
        a, v = model(mel[None], im[None])
        print(a.shape, v.shape)
    # test_dataset = SyncDataset("val")
    exit()

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=hparams.syncnet_batch_size,
        shuffle=True,
        num_workers=8,
    )

    # test_data_loader = DataLoader(
    #     test_dataset, batch_size=hparams.syncnet_batch_size, num_workers=8
    # )
    test_data_loader = None

    device = torch.device("cuda" if use_cuda else "cpu")

    # Model
    model = SyncNet().to(device)
    print(
        "total trainable params {}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )

    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=hparams.syncnet_lr
    )

    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False)

    train(
        device,
        model,
        train_data_loader,
        test_data_loader,
        optimizer,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=hparams.syncnet_checkpoint_interval,
        nepochs=hparams.nepochs,
    )
