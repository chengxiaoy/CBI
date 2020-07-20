import warnings
import warnings
warnings.filterwarnings("ignore")
import callbacks as clb
import configuration as C
import model as models
import utils as utils
from sklearn.model_selection import StratifiedKFold
from catalyst.dl import SupervisedRunner

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
from criterion import ResNetLoss
from model import build_model

df = pd.read_csv('./data/resampled_train.csv')
skf = StratifiedKFold(n_splits=5, random_state=33, shuffle=True)
splits = skf.split(df, y=df['ebird_code'].values)

melspectrogram_parameters = {
    "n_mels": 128,
    "fmin": 20,
    "fmax": 16000
}

model_config = {
    "base_model_name": "resnet50",
    "pretrained": True,
    "num_classes": 264
}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@timer(name="train_model", logger=None)
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    train_loader = dataloader
    for batch_idx, (img_batch, labels) in enumerate(tqdm(train_loader)):
        img_batch = img_batch.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model(img_batch)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        loss_value = loss.item()
        # print(loss_value)
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
    epoch_loss = epoch_loss / len(val_dataloader)
    return epoch_loss


def train(index, model, dataloaders, optimizer, criterion, scheduler, n_epoch, device):
    min_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(n_epoch):
        train_loss = train_model(model, dataloaders['train'], optimizer, criterion, device)
        val_loss = evaluate_model(model, dataloaders['val'], criterion, device)
        print("train_loss:{} val_loss:{}".format(train_loss, val_loss))
        if val_loss < min_loss:
            min_loss = val_loss
            torch.save(model.state_dict(), '{}_best_model.pth'.format(index))
            best_model_wts = copy.deepcopy(model.state_dict())
        scheduler.step(val_loss)

    model.load_state_dict(best_model_wts)
    return model


for i, (train_index, val_index) in enumerate(splits):
    train_df = df.loc[train_index, :].reset_index(drop=True)
    val_df = df.loc[val_index, :].reset_index(drop=True)
    train_dataset = SpectrogramDataset(train_df, datadir=Path('./data/train_audio_resampled'),
                                       melspectrogram_parameters=melspectrogram_parameters)
    val_dataset = SpectrogramDataset(val_df, datadir=Path('./data/train_audio_resampled'),
                                     melspectrogram_parameters=melspectrogram_parameters)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, num_workers=8)
    dataloaders = {"train": train_dataloader, "val": val_dataloader}
    model = build_model(model_config)
    criterion = ResNetLoss(loss_type="bce")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    train(i, model, dataloaders, optimizer, criterion, lr_scheduler, 50, device)
