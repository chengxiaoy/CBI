import warnings
import warnings

warnings.filterwarnings("ignore")
import callbacks as clb
import configuration as C
import model as models
import utils as utils
from sklearn.model_selection import StratifiedKFold
from catalyst.dl import SupervisedRunner
from config import Config
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from cbi_data import SpectrogramDataset
from torch.utils.data import DataLoader
from fastprogress import progress_bar
import torch
from utils import timer
import copy
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from criterion import get_loss
from model import build_model
import random
import numpy as np
import os
from tensorboardX import SummaryWriter


@timer(name="train_model", logger=None)
def train_model(model, dataloader, optimizer, criterion, scheduler, config: Config):
    model.train()
    epoch_loss = 0
    train_loader = dataloader
    for batch_idx, (img_batch, labels) in enumerate(tqdm(train_loader)):
        img_batch = img_batch.to(config.device)
        labels = labels.to(config.device)
        if config.use_half:
            img_batch = img_batch.half()
        optimizer.zero_grad()
        output = model(img_batch)
        # if config.use_half:
        #     output['logits'] = output['logits'].float()
        #     output['multiclass_proba'] = output['multiclass_proba'].float()
        #     output['logits'] = output['logits'].float()
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        if config.scheduler_type == 'cos':
            scheduler.step()
        loss_value = loss.item()
        epoch_loss += loss_value
    epoch_loss = epoch_loss / len(train_loader)
    return epoch_loss


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    val_loader = dataloader
    with torch.no_grad():
        for batch_idx, (img_batch, labels) in enumerate(progress_bar(val_loader)):
            img_batch = img_batch.to(device)
            labels = labels.to(device)
            output = model(img_batch)
            loss = criterion(output, labels)
            epoch_loss += loss.item()
    epoch_loss = epoch_loss / len(val_loader)
    return epoch_loss


def train(index, model, dataloaders, optimizer, criterion, scheduler, writer, config: Config):
    min_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(config.N_EPOCH):
        train_loss = train_model(model, dataloaders['train'], optimizer, criterion, scheduler, config)
        val_loss = evaluate_model(model, dataloaders['val'], criterion, config.device)
        print("train_loss:{} val_loss:{}".format(train_loss, val_loss))
        if val_loss < min_loss:
            min_loss = val_loss
            torch.save(model.state_dict(),
                       './checkpoints/exp_{}_index_{}_best_model.pth'.format(config.expriment_id, index))
            best_model_wts = copy.deepcopy(model.state_dict())
        if config.scheduler_type == 'Plateau':
            scheduler.step(val_loss)
        writer.add_scalars('cv_{}/loss'.format(index), {'train': train_loss, 'val': val_loss},
                           epoch)
    model.load_state_dict(best_model_wts)
    return model


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


def training(config: Config):
    set_seed(33)
    df = pd.read_csv('./data/resampled_train.csv')
    skf = StratifiedKFold(n_splits=config.split_n, random_state=33, shuffle=True)
    splits = skf.split(df, y=df['ebird_code'].values)

    writer = SummaryWriter(logdir=os.path.join("./board/", str(config.expriment_id)))

    melspectrogram_parameters = {
        "n_mels": 128,
        "fmin": 20,
        "fmax": 16000
    }
    for i, (train_index, val_index) in enumerate(splits):
        train_df = df.loc[train_index, :].reset_index(drop=True)
        val_df = df.loc[val_index, :].reset_index(drop=True)
        train_dataset = SpectrogramDataset(train_df, datadir=Path('./data/train_audio_resampled'),
                                           melspectrogram_parameters=melspectrogram_parameters)
        val_dataset = SpectrogramDataset(val_df, datadir=Path('./data/train_audio_resampled'),
                                         melspectrogram_parameters=melspectrogram_parameters)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                                      num_workers=config.NUM_WORKER)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=48, shuffle=False,
                                    num_workers=config.NUM_WORKER)
        dataloaders = {"train": train_dataloader, "val": val_dataloader}
        model = build_model(config)
        criterion = get_loss(config.loss_type)
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        lr_scheduler = None
        if config.scheduler_type == 'Plateau':
            lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        train(i, model, dataloaders, optimizer, criterion, lr_scheduler, writer=writer, config=config)


if __name__ == '__main__':
    config = Config()
    config.model_name = 'efficientnet-b0'
    training(config)
