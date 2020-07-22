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

from sklearn.metrics import f1_score, accuracy_score


@timer(name="train_model", logger=None)
def train_model(model, dataloader, optimizer, criterion, scheduler, config: Config):
    model.train()
    epoch_loss = 0
    train_loader = dataloader
    train_preds, train_true = torch.Tensor([]).to(config.device), torch.LongTensor([]).to(config.device)
    for batch_idx, (img_batch, labels) in enumerate(tqdm(train_loader)):
        img_batch = img_batch.to(config.device)
        labels = labels.to(config.device)
        if config.use_half:
            img_batch = img_batch.half()
        optimizer.zero_grad()
        output = model(img_batch)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        if config.scheduler_type == 'cos' or config.scheduler_type == 'cyc':
            scheduler.step()
        loss_value = loss.item()
        epoch_loss += loss_value

        train_true = torch.cat([train_true, labels], 0)
        train_preds = torch.cat([train_preds, output['multilabel_proba']], 0)

    train_true = train_true.cpu().detach().numpy()
    a = np.zeros(train_preds.shape)
    train_preds_index = train_preds.cpu().detach().numpy() > 0.5
    a[train_preds_index] = 1
    train_score = f1_score(train_true, a, average='macro')
    epoch_loss = epoch_loss / len(train_loader)

    return epoch_loss, train_score


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    val_loader = dataloader
    val_preds, val_true = torch.Tensor([]).to(config.device), torch.LongTensor([]).to(config.device)
    with torch.no_grad():
        for batch_idx, (img_batch, labels) in enumerate(progress_bar(val_loader)):
            img_batch = img_batch.to(device)
            labels = labels.to(device)
            output = model(img_batch)
            loss = criterion(output, labels)
            epoch_loss += loss.item()

            val_true = torch.cat([val_true, labels], 0)
            val_preds = torch.cat([val_preds, output['multilabel_proba']], 0)

    epoch_loss = epoch_loss / len(val_loader)
    val_true = val_true.cpu().detach().numpy()
    b = np.zeros(val_preds.shape)
    val_preds_index = val_preds.cpu().detach().numpy() > 0.5
    b[val_preds_index] = 1
    val_score = f1_score(val_true, b, average='macro')

    return epoch_loss, val_score


def train(index, model, dataloaders, optimizer, criterion, scheduler, writer, config: Config):
    min_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(config.N_EPOCH):
        train_loss, train_score = train_model(model, dataloaders['train'], optimizer, criterion, scheduler, config)
        val_loss, val_score = evaluate_model(model, dataloaders['val'], criterion, config.device)
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

        writer.add_scalars('cv_{}/score'.format(index), {'train': train_score, 'val': val_score},
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
        criterion = get_loss(config)
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        lr_scheduler = None
        if config.scheduler_type == 'Plateau':
            lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

        if config.scheduler_type == 'cos':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader))

        if config.scheduler_type == 'cyc':
            lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.001,
                                                             step_size_up=len(train_dataloader) // 2,
                                                             cycle_momentum=False)
        train(i, model, dataloaders, optimizer, criterion, lr_scheduler, writer=writer, config=config)


if __name__ == '__main__':
    # config = Config()
    # config.model_name = 'efficientnet-b0'
    # training(config)

    # config = Config()
    # config.expriment_id = 2
    # config.N_EPOCH = 30
    # config.model_name = 'efficientnet-b2'
    # training(config)

    # config = Config()
    # config.expriment_id = 3
    # config.N_EPOCH = 50
    # config.model_name = 'efficientnet-b0'
    # config.loss_type = 'focal'
    # training(config)

    # config = Config()
    # config.expriment_id = 4
    # config.N_EPOCH = 50
    # config.model_name = 'efficientnet-b0'
    # config.loss_type = 'ce'
    # training(config)

    # config = Config()
    # config.expriment_id = 5
    # config.N_EPOCH = 50
    # config.model_name = 'efficientnet-b0'
    # config.loss_type = 'bce'
    # training(config)
    #
    # config = Config()
    # config.expriment_id = 6
    # config.N_EPOCH = 50
    # config.model_name = 'efficientnet-b0'
    # config.loss_type = 'focal'
    # training(config)

    config = Config()
    config.expriment_id = 7
    config.N_EPOCH = 50
    config.model_name = 'efficientnet-b0'
    config.loss_type = 'focal'
    config.scheduler_type = 'cyc'
    training(config)
