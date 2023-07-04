import os.path as osp
from pathlib import Path
from tqdm import tqdm
from contextlib import contextmanager
from torch.utils.data import Dataset, DataLoader, distributed
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from models import SyncNet_color as SyncNet
from models import Wav2Lip as Wav2Lip
import audio

import torch
from torch import nn
from torch import optim
import numpy as np

from glob import glob

import os
import random
import cv2
from hparams import hparams

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))

global_step = 0
epoch = 0
window_size = 5
syncnet_mel_step_size = 16


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """Decorator to make all processes in distributed training wait for each local_master to do something."""
    initialized = torch.distributed.is_available() and torch.distributed.is_initialized()
    if initialized and local_rank not in (-1, 0):
        dist.barrier(device_ids=[local_rank])
    yield
    if initialized and local_rank == 0:
        dist.barrier(device_ids=[0])


class Wav2LipDataset(Dataset):
    """Wav2LipDataset

    Attributes:
        root
        └── images
            └── id
                └── 0.jpg
                └── 1.jpg
                └── ...
                └── n.jpg
        └── audios
            └── id.wav/aac
    """

    def __init__(self, im_dir, audio_dir):
        print("Loading image files...")
        id_dirs = glob(osp.join(im_dir, "*"))
        self.im_files = []
        self.im_range = {}
        print("Checking image files...")
        for id_dir in id_dirs:
            frame_files = glob(osp.join(id_dir, "*"))
            frame_ids = [self.get_frame_id(frame_file) for frame_file in frame_files]
            # NOTE: filter frames that frame id less than 2, for `self.get_segmented_mels`
            frame_files = [
                frame_file for i, frame_file in enumerate(frame_files) if frame_ids[i] > 2
            ]
            frame_ids = [frame_id for frame_id in frame_ids if frame_id > 2]
            # NOTE: check the numbers of each id, should be more than `window_size * 2`
            if len(frame_files) < 2 * window_size:
                print(
                    f"WARNING: The number of frames should be more than {2 * window_size}, \
                        but got {len(frame_files)} for {id_dir}, ignoring {id_dir}."
                )
                continue
            self.im_files += frame_files
            self.im_range[Path(id_dir).name] = (min(frame_ids), max(frame_ids) + 1)
        print(f"Loaded {len(self.im_files)} image files...")

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

    def read_window(self, window_fnames):
        if window_fnames is None:
            return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                img = cv2.resize(img, (hparams.img_size, hparams.img_size))
            except Exception as e:
                return None

            window.append(img)

        return window

    def crop_audio_window(self, mel, frame_id, neg_sample=False):
        start_idx = int(80.0 * (frame_id / float(hparams.fps)))
        mel_len = len(mel)
        if neg_sample:
            idx = random.randint(0, mel_len - syncnet_mel_step_size)
            while start_idx == idx:
                idx = random.randint(0, mel_len - syncnet_mel_step_size)
            start_idx = idx
        end_idx = start_idx + syncnet_mel_step_size
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

    def get_segmented_mels(self, spec, frame_id):
        mels = []
        assert window_size == 5
        # (1, 2, 3, 4, 5)  -> (-1, 0, 1, 2, 3)
        start_id = frame_id + 1  # 0-indexing ---> 1-indexing
        for i in range(start_id, start_id + window_size):
            m = self.crop_audio_window(spec, i - 2)
            mels.append(m.T)

        mels = np.asarray(mels)

        return mels

    def prepare_window(self, window):
        # 3 x T x H x W
        x = np.asarray(window)
        x = np.transpose(x, (3, 0, 1, 2))
        return x

    def __len__(self):
        return len(self.im_files)

    def __getitem__(self, idx):
        im_file = self.im_files[idx]
        p = Path(im_file)
        id = p.parent.name
        mel = self.audios[id]
        window, frame_id = self.generate_window(p)

        wrong_frame_id = random.choice(range(*self.im_range[id]))
        wrong_im_file = p.parent / f"{wrong_frame_id}.jpg"
        while (wrong_frame_id == frame_id) or (not wrong_im_file.exists()):
            wrong_frame_id = random.choice(range(*self.im_range[id]))
            wrong_im_file = p.parent / f"{wrong_frame_id}.jpg"
        wrong_window, wrong_frame_id = self.generate_window(wrong_im_file)

        mel_patch = self.crop_audio_window(mel.copy(), frame_id)
        indiv_mels = self.get_segmented_mels(mel.copy(), frame_id)

        window = self.prepare_window(window)
        y = window.copy()
        window[:, :, window.shape[2] // 2 :] = 0.0

        wrong_window = self.prepare_window(wrong_window)
        x = np.concatenate([window, wrong_window], axis=0)

        x = torch.from_numpy(x)
        mel_patch = torch.from_numpy(mel_patch.T).unsqueeze(0)
        indiv_mels = torch.from_numpy(indiv_mels).unsqueeze(1)
        y = torch.from_numpy(y)
        return x, indiv_mels, mel_patch, y


def save_sample_images(x, g, gt, epoch, checkpoint_dir):
    x = (x.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.0).astype(np.uint8)
    g = (g.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.0).astype(np.uint8)
    gt = (gt.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.0).astype(np.uint8)

    refs, inps = x[..., 3:], x[..., :3]
    folder = osp.join(checkpoint_dir, "samples_step{:09d}".format(epoch))
    if not osp.exists(folder):
        os.mkdir(folder)
    collage = np.concatenate((refs, inps, g, gt), axis=-2)
    for batch_idx, c in enumerate(collage):
        for t in range(len(c)):
            cv2.imwrite("{}/{}_{}.jpg".format(folder, batch_idx, t), c[t])


logloss = nn.BCELoss()


def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v).clamp_(0, 1)
    loss = logloss(d.unsqueeze(1), y)

    return loss


# NOTE: set devices
# device = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = device  # set environment variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if LOCAL_RANK != -1:
    assert torch.cuda.device_count() > LOCAL_RANK, "insufficient CUDA devices for DDP command"
    torch.cuda.set_device(LOCAL_RANK)
    device = torch.device("cuda", LOCAL_RANK)
    dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

recon_loss = nn.L1Loss()


def get_sync_loss(mel, g):
    g = g[:, :, :, g.size(3) // 2 :]
    g = torch.cat([g[:, :, i] for i in range(window_size)], dim=1)
    # B, 3 * T, H//2, W
    a, v = syncnet(mel, g)
    y = torch.ones(g.size(0), 1).float().to(device)
    return cosine_loss(a, v, y)


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
        if epoch == 20:
            # without image GAN a lesser weight is sufficient
            hparams.set_hparam("syncnet_wt", 0.01)

        running_sync_loss, running_l1_loss = 0.0, 0.0
        prog_bar = enumerate(train_loader)
        if RANK in (-1, 0):
            prog_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (x, indiv_mels, mel, gt) in prog_bar:
            n = epoch * nb + i
            model.train()
            optimizer.zero_grad()

            # Move data to CUDA device
            x = x.to(device).float() / 255.0
            gt = gt.to(device).float() / 255.0
            mel = mel.to(device).float()
            indiv_mels = indiv_mels.to(device).float()

            g = model(indiv_mels, x)

            sync_loss = get_sync_loss(mel, g) if hparams.syncnet_wt > 0.0 else 0.0
            l1loss = recon_loss(g, gt)
            if RANK != -1:
                sync_loss *= WORLD_SIZE
                l1loss *= WORLD_SIZE

            loss = hparams.syncnet_wt * sync_loss + (1 - hparams.syncnet_wt) * l1loss
            loss.backward()
            optimizer.step()

            running_l1_loss += l1loss.item()
            if hparams.syncnet_wt > 0.0:
                running_sync_loss += sync_loss.item()
            else:
                running_sync_loss += 0.0

            if RANK in (0, -1):
                prog_bar.set_description(
                    "Epoch: {}, L1: {}, Sync Loss: {}".format(
                        epoch, running_l1_loss / (i + 1), running_sync_loss / (i + 1)
                    )
                )

        if RANK in (-1, 0):
            save_sample_images(x, g, gt, epoch, checkpoint_dir)
            save_checkpoint(model, optimizer, n, checkpoint_dir, epoch)
            with torch.no_grad():
                average_sync_loss = eval_model(val_loader, device, model)

            # if average_sync_loss < 0.75:
            #     # without image GAN a lesser weight is sufficient
            #     hparams.set_hparam("syncnet_wt", 0.01)
        # NOTE: This barrier() here gets stuck while DDP training
        # dist.barrier()
        # dist.broadcast(average_sync_loss, src=0)
        # NOTE: scatter `syncnet_wt` to other machines.
        # out_syncnet_wt = [0.0]   # it has to be a list object.
        # dist.scatter_object_list(out_syncnet_wt, [hparams.syncnet_wt for _ in range(WORLD_SIZE)], src=0)
        # hparams.set_hparam("syncnet_wt", out_syncnet_wt[0])


def eval_model(val_loader, device, model):
    eval_steps = 700
    sync_losses, recon_losses = [], []
    step = 0
    pbar = tqdm(val_loader, total=len(val_loader))
    for x, indiv_mels, mel, gt in pbar:
        step += 1
        model.eval()

        # Move data to CUDA device
        x = x.to(device).float() / 255.0
        gt = gt.to(device).float() / 255.0
        mel = mel.to(device).float()
        indiv_mels = indiv_mels.to(device).float()

        g = model(indiv_mels, x)

        sync_loss = get_sync_loss(mel, g)
        l1loss = recon_loss(g, gt)

        sync_losses.append(sync_loss.item())
        recon_losses.append(l1loss.item())
        pbar.set_description("Evaluating for {} steps".format(eval_steps))

        if step > eval_steps:
            break
    averaged_sync_loss = sum(sync_losses) / len(sync_losses)
    averaged_recon_loss = sum(recon_losses) / len(recon_losses)

    print("L1: {}, Sync loss: {}".format(averaged_recon_loss, averaged_sync_loss))

    return averaged_sync_loss


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


def load_checkpoint(path, model, optimizer, reset_optimizer=False):
    print("Load checkpoint from: {}".format(path))
    checkpoint = torch.load(path, map_location="cpu")
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace("module.", "")] = v
    model.load_state_dict(new_s)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])

    return model


if __name__ == "__main__":
    checkpoint_dir = "runs/wav2lip_ddp"
    checkpoint_path = None
    syncnet_checkpoint_path = "./runs/syncnet_first/checkpoint_step000490000.pth"

    # Dataset and Dataloader setup
    with torch_distributed_zero_first(RANK):
        train_dataset = Wav2LipDataset(
            im_dir="/data/datasets/audio/RD25_images",
            audio_dir="/data/datasets/audio/RD25_audios/npy",
        )

    sampler = None if RANK == -1 else distributed.DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=hparams.batch_size,
        shuffle=sampler is None,
        num_workers=8,
        sampler=sampler,
    )

    val_loader = None
    if RANK in (-1, 0):
        # val_dataset = Wav2LipDataset(
        #     im_dir="/d/dataset/audio/HDTF_DATA/RD25_images",
        #     audio_dir="/d/dataset/audio/HDTF_DATA/RD25_audios/npy",
        # )
        val_loader = DataLoader(train_dataset, batch_size=hparams.batch_size, num_workers=4)

    # Model
    model = Wav2Lip().to(device)
    if RANK in (0, -1):
        print(
            "total trainable params {}".format(
                sum(p.numel() for p in model.parameters() if p.requires_grad)
            )
        )
    if WORLD_SIZE > 1:
        model = DDP(model, device_ids=[RANK])
    # model.eval()
    # for i, (im, indiv_mels, mel, gt) in enumerate(train_dataset):
    #     # print(i, x.shape, mel.shape, y)
    #     im = im.to(device).float() / 255.0
    #     gt = gt.to(device).float() / 255.0
    #     mel = mel.to(device).float()
    #     indiv_mels = indiv_mels.to(device).float()
    #     g = model(indiv_mels[None], im[None])
    #     print(g.shape)
    # exit()

    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=hparams.initial_learning_rate
    )

    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False)

    syncnet = SyncNet()
    for p in syncnet.parameters():
        p.requires_grad = False

    load_checkpoint(
        syncnet_checkpoint_path,
        syncnet,
        None,
        reset_optimizer=True,
    )
    syncnet.to(device)
    syncnet.eval()

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Train!
    train(
        device,
        model,
        train_loader,
        val_loader,
        optimizer,
        checkpoint_dir=checkpoint_dir,
        epochs=hparams.epochs,
    )
