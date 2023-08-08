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


use_cuda = torch.cuda.is_available()
print("use_cuda: {}".format(use_cuda))

# NOTE: video fps is 25 and audio fps is 80, so 5 frames for video corresponding to 16 frames for audio
window_size = 5
syncnet_mel_step_size = 16


class SyncDataset(Dataset):
    """SyncDataset

    Attributes:
    root1
    └── root2
        └── images
            └── id
                └── *.jpg
        └── audios
            └── id.wav/aac
    """

    def __init__(self, im_dir, audio_dir):
        print("Loading image files...")
        id_dirs = glob(osp.join(im_dir, "*", "*"))
        self.im_files = []
        print("Checking image files...")
        for id_dir in id_dirs:
            frame_files = glob(osp.join(id_dir, "*"))
            frame_ids = [self.get_frame_id(frame_file) for frame_file in frame_files]
            if sum(frame_ids) != sum(range(min(frame_ids), max(frame_ids) + 1)):
                print("WARNING: The numbers of frames should be continuous, "
                      f"but got discontinuous numbers, ignoring {id_dir}.")
                continue
            # NOTE: check the numbers of each id, should be more than `window_size * 2`
            if len(frame_files) < 2 * window_size:
                print(f"WARNING: The number of frames should be more than {2 * window_size}, "
                        f"but got {len(frame_files)} for {id_dir}, ignoring {id_dir}.")
                continue
            self.im_files += frame_files
        print(f"Loaded {len(self.im_files)} image files...")
        print("Loading audios...")
        audios = {}
        audio_files = glob(osp.join(audio_dir, "*", "*"))
        for af in tqdm(audio_files, total=len(audio_files)):
            # mel = self.get_mel(af)
            # audios[Path(af).with_suffix("").name] = mel
            # np.save(af.replace(".aac", ".npy"), mel)
            audios[str(Path(af).with_suffix(""))] = np.load(af)
        self.audios = audios

    def get_frame_id(self, im_file):
        return int(Path(im_file).with_suffix("").name)

    def get_mel(self, audio_file):
        wav = audio.load_wav(audio_file, hparams.sample_rate)
        mel = audio.melspectrogram(wav).T
        return mel

    def crop_audio_window(self, mel, frame_id, neg_sample=False):
        start_idx = int(80.0 * (frame_id / float(hparams.fps)))
        mel_len = len(mel)
        if neg_sample:
            idx = random.randint(0, mel_len - syncnet_mel_step_size)
            while start_idx == idx:
                idx = random.randint(0, mel_len - syncnet_mel_step_size)
            start_idx = idx
        end_idx = start_idx + syncnet_mel_step_size
        # TODO
        # NOTE: handle the case that `end_idx` beyond the length of mel.
        if end_idx >= mel_len:
            start_idx = mel_len - syncnet_mel_step_size
            end_idx = mel_len
        return mel[start_idx:end_idx, :]

    def generate_window(self, p):
        window = []
        frame_id = int(p.with_suffix("").name)
        end_id = frame_id + window_size
        end_exist = (p.parent / f"{end_id}.jpg").exists()
        iterator = range(frame_id, end_id) if end_exist else range(frame_id - window_size, frame_id)
        frame_id = frame_id if end_exist else (frame_id - window_size)
        for fname in [str(p.parent / f"{i}.jpg") for i in iterator]:
            im = cv2.imread(fname)
            im = cv2.resize(im, (hparams.img_size, hparams.img_size))
            window.append(im)
        return window, frame_id

    def __len__(self):
        return len(self.im_files)

    def __getitem__(self, idx):
        im_file = self.im_files[idx]
        p = Path(im_file)
        akey = str(p.parent).replace("images", "audios")
        mel = self.audios[akey]
        window, frame_id = self.generate_window(p)

        neg_sample = random.uniform(0, 1) < 0.5
        mel_patch = self.crop_audio_window(mel.copy(), frame_id, neg_sample=neg_sample)
        y = torch.tensor([0] if neg_sample else [1], dtype=torch.float32)

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
    epochs=None,
):
    nb = len(train_loader)  # number of batches
    for epoch in range(epochs):
        running_loss = 0.0
        prog_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (im, mel, y) in prog_bar:
            n = epoch * nb + i
            model.train()
            optimizer.zero_grad()

            # Transform data to CUDA device
            im = im.to(device).float() / 255.0
            mel = mel.to(device).float()
            y = y.to(device)

            # (b, 1, 80, 16), (b, 15, 128, 256)
            a, v = model(mel, im)
            loss = cosine_loss(a, v, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.detach().cpu().item()

            prog_bar.set_description("Loss: {}".format(running_loss / (i + 1)))


        save_checkpoint(model, optimizer, n, checkpoint_dir, epoch)
        with torch.no_grad():
            eval_model(val_loader, device, model)


def eval_model(val_loader, device, model):
    print("Evaluating...")
    losses = []
    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for step, (im, mel, y) in pbar:
        model.eval()

        # Transform data to CUDA device
        im = im.to(device).float() / 255.0
        mel = mel.to(device).float()

        a, v = model(mel, im)
        y = y.to(device)

        loss = cosine_loss(a, v, y)
        losses.append(loss.item())

    averaged_loss = sum(losses) / len(losses)
    print(averaged_loss)


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):
    checkpoint_path = osp.join(checkpoint_dir, "checkpoint_step{:09d}.pth".format(epoch))
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
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])

    return model


if __name__ == "__main__":
    checkpoint_dir = "runs/syncnet"
    checkpoint_path = None

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    device = torch.device("cuda:1" if use_cuda else "cpu")
    # Model
    model = SyncNet().to(device)
    print(
        "total trainable params {}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )
    # Dataset and Dataloader setup
    train_dataset = SyncDataset(
        im_dir="/sdata/datasets/audio/final/train/images",
        audio_dir="/sdata/datasets/audio/final/train/audios",
    )
    # model.eval()
    # for i, (im, mel, y, imf) in enumerate(train_dataset):
    #     print(i, im.shape, mel.shape, y, imf)
    #     assert im.shape == (15, 128, 256), f"{im.shape}"
    #     assert mel.shape == (1, 80, 16), f"{mel.shape}"
    #     # im = im.to(device).float() / 255.0
    #     # mel = mel.to(device).float()
    #     # a, v = model(mel[None], im[None])
    #     # print(a.shape, v.shape)
    # exit()

    val_dataset = SyncDataset(
        im_dir="/sdata/datasets/audio/final/val/images",
        audio_dir="/sdata/datasets/audio/final/val/audios",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=hparams.syncnet_batch_size,
        shuffle=True,
        num_workers=16,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=hparams.syncnet_batch_size * 2, num_workers=8
    )

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
        epochs=hparams.epochs,
    )
