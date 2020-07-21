import torch.nn as nn


class ResNetLoss(nn.Module):
    def __init__(self, loss_type="ce"):
        super().__init__()

        self.loss_type = loss_type
        if loss_type == "ce":
            self.loss = nn.CrossEntropyLoss()
        elif loss_type == "bce":
            self.loss = nn.BCELoss()

    def forward(self, input, target):
        if self.loss_type == "ce":
            input_ = input["logits"]
            target = target.argmax(1).long()
        elif self.loss_type == "bce":
            input_ = input["multilabel_proba"]
            target = target.float()
            # assert (input_.data.cpu().numpy().all() >= 0. and input_.data.cpu().numpy().all() <= 1.)

        return self.loss(input_, target)


def get_loss(loss_type):
    if loss_type == 'ce':
        return ResNetLoss(loss_type='ce')
    if loss_type == 'bce':
        return ResNetLoss(loss_type='bce')
    if loss_type == 'focal':
        return None
    if loss_type == 'arcface':
        return None
    if loss_type == 'A-softmax':
        return None
