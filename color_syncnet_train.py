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

# NOTE: video fps is 25 and audio fps is 80, so 5 frames for video corresponding to 16 frames for audio
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

    def __init__(self, im_dir, audio_dir):
        print("Loading image files...")
        self.im_files = glob(osp.join(im_dir, "*", "*"))
        print(f"Loaded {len(self.im_files)} image files...")
        # NOTE: check the numbers of each id, should be more than `window_size * 2`
        print("Checking image files...")
        for im_file in self.im_files:
            assert self.get_frame_id(im_file) > 2 * window_size
        print("Loading audios...")
        audios = {}
        audio_files = glob(osp.join(audio_dir, "*"))
        for af in tqdm(audio_files, total=len(audio_files)):
            # mel = self.get_mel(af)
            # audios[Path(af).with_suffix("").name] = mel
            # np.save(af.replace(".aac", ".npy"), mel)
            audios[Path(af).with_suffix("").name] = np.load(af)
        self.audios = audios

    def get_frame_id(self, im_file):
        return int(Path(im_file).with_suffix("").name)

    def get_mel(self, audio_file):
        wav = audio.load_wav(audio_file, hparams.sample_rate)
        mel = audio.melspectrogram(wav).T
        return mel

    def crop_audio_window(self, mel, frame_id, neg_sample=False):
        start_idx = int(80.0 * (frame_id / float(hparams.fps)))
        if neg_sample:
            idx = random.randint(0, len(mel) - syncnet_mel_step_size)
            while start_idx == idx:
                idx = random.randint(0, len(mel) - syncnet_mel_step_size)
            start_idx = idx
        end_idx = start_idx + syncnet_mel_step_size
        return mel[start_idx:end_idx, :]

    def generate_window(self, p):
        frame_id = int(p.with_suffix("").name)
        end_id = frame_id + window_size
        end_exist = (p.parent / f"{end_id}.jpg").exists()
        iterator = range(frame_id, end_id) if end_exist else range(frame_id - window_size, frame_id)
        frame_id = frame_id if end_exist else (frame_id - window_size)
        return [str(p.parent / f"{i}.jpg") for i in iterator], frame_id

    def __len__(self):
        return len(self.im_files)

    def __getitem__(self, idx):
        im_file = self.im_files[idx]
        p = Path(im_file)
        mel = self.audios[p.parent.name]
        window_files, frame_id = self.generate_window(p)

        window = []
        for fname in window_files:
            im = cv2.imread(fname)
            im = cv2.resize(im, (hparams.img_size, hparams.img_size))
            window.append(im)

        neg_sample = idx % 2
        mel_patch = self.crop_audio_window(mel.copy(), frame_id, neg_sample=neg_sample)
        y = torch.ones(0 if neg_sample else 1, dtype=torch.float32)

        # H x W x 3 * T
        x = np.concatenate(window, axis=2)
        x = x.transpose(2, 0, 1)  # 3 * T, H, W
        x = x[:, x.shape[1] // 2 :]  # 3 * T, H // 2, W

        x = torch.from_numpy(x)
        mel_patch = torch.from_numpy(mel_patch.T).unsqueeze(0)

        return x, mel_patch, y


logloss = nn.BCELoss()


def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v).clamp_(0, 1)
    # d = (a * v).sum(-1)
    loss = logloss(d.unsqueeze(1), y)

    return loss


def train(
    device,
    model,
    train_loader,
    val_loader,
    optimizer,
    checkpoint_dir=None,
    checkpoint_interval=None,
    nepochs=None,
):
    global global_step, global_epoch

    while global_epoch < nepochs:
        running_loss = 0.0
        prog_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for step, (im, mel, y) in prog_bar:
            if im is None:
                continue
            model.train()
            optimizer.zero_grad()

            # Transform data to CUDA device
            im = im.to(device).float() / 255.0
            mel = mel.to(device).float()
            y = y.to(device)

            a, v = model(mel, im)
            loss = cosine_loss(a, v, y)
            loss.backward()
            optimizer.step()

            global_step += 1
            running_loss += loss.detach().cpu().item()

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
        for step, (im, mel, y) in enumerate(test_data_loader):
            model.eval()

            # Transform data to CUDA device
            im = im.to(device).float() / 255.0
            mel = mel.to(device).float()

            a, v = model(mel, im)
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
    print(
        "total trainable params {}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )
    # Dataset and Dataloader setup
    train_dataset = SyncDataset(
        im_dir="/d/dataset/audio/HDTF_DATA/RD25_images",
        audio_dir="/d/dataset/audio/HDTF_DATA/RD25_audios/npy",
    )
    # model.eval()
    # for i, (im, mel, y) in enumerate(train_dataset):
    #     # print(i, x.shape, mel.shape, y)
    #     im = im.to(device).float() / 255.0
    #     mel = mel.to(device).float()
    #     a, v = model(mel[None], im[None])
    #     print(a.shape, v.shape)
    # exit()

    # val_dataset = SyncDataset("val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=hparams.syncnet_batch_size,
        shuffle=True,
        num_workers=8,
    )

    # val_loader = DataLoader(
    #     val_dataset, batch_size=hparams.syncnet_batch_size, num_workers=8
    # )
    val_loader = None

    device = torch.device("cuda" if use_cuda else "cpu")

    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=hparams.syncnet_lr
    )

    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False)

    train(
        device,
        model,
        train_loader,
        val_loader,
        optimizer,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=hparams.syncnet_checkpoint_interval,
        nepochs=hparams.nepochs,
    )
