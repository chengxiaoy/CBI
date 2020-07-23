import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
from config import Config
from efficientnet_pytorch import EfficientNet


class ResNet(nn.Module):
    def __init__(self, base_model_name: str, pretrained=False,
                 num_classes=264):
        super().__init__()
        base_model = models.__getattribute__(base_model_name)(
            pretrained=pretrained)
        layers = list(base_model.children())[:-2]
        layers.append(nn.AdaptiveMaxPool2d(1))
        self.encoder = nn.Sequential(*layers)

        in_features = base_model.fc.in_features

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 1024), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(1024, num_classes))

    def forward(self, x):
        batch_size = x.size(0)
        x = self.encoder(x).view(batch_size, -1)
        x = self.classifier(x)
        multiclass_proba = F.softmax(x, dim=1)
        multilabel_proba = torch.sigmoid(x)
        return {
            "logits": x,
            "multiclass_proba": multiclass_proba,
            "multilabel_proba": multilabel_proba
        }


class EffModel(nn.Module):
    def __init__(self, base_model_name, num_classes):
        super(EffModel, self).__init__()
        self.base_model = EfficientNet.from_pretrained(base_model_name)
        self.mp = nn.AdaptiveMaxPool2d(1)
        in_features = self.base_model._fc.in_features
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 1024), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(1024, num_classes))

    def forward(self, x):
        batch_size = x.size(0)
        x = self.base_model.extract_features(x)
        x = self.mp(x).view(batch_size, -1)
        x = self.classifier(x)
        multiclass_proba = F.softmax(x, dim=1)
        multilabel_proba = torch.sigmoid(x)
        return {
            "logits": x,
            "multiclass_proba": multiclass_proba,
            "multilabel_proba": multilabel_proba
        }


def build_model(config: Config):
    model = None
    if config.model_name.startswith("resnet"):
        model = ResNet(config.model_name, True, config.N_CLASS)
    elif config.model_name.startswith("eff"):
        model = EffModel(config.model_name, config.N_CLASS)
    if config.use_half:
        model.half()
    model.to(config.device)
    model.train()
    return model


def get_model(config: Config, weights_path: str):
    model = build_model(config)
    checkpoint = torch.load(weights_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = config.device
    model.to(device)
    model.eval()
    return model
