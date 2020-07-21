import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import Parameter
from config import Config


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


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        targets = targets.float()
        inputs = inputs["multilabel_proba"]

        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)

        alpha = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        F_loss = alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class Am_softmax(nn.Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599
    def __init__(self, embedding_size=512, classnum=51332):
        super(Am_softmax, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size, classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.m = 0.35  # additive margin recommended by the paper
        self.s = 30.  # see normface https://arxiv.org/abs/1704.06369

    def forward(self, embbedings, label):
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        phi = cos_theta - self.m
        label = label.view(-1, 1)  # size=(B,1)
        index = cos_theta.data * 0.0  # size=(B,Classnum)
        index.scatter_(1, label.data.view(-1, 1), 1)
        index = index.byte()
        output = cos_theta * 1.0
        output[index] = phi[index]  # only change the correct predicted output
        output *= self.s  # scale up in order to make softmax work, first introduced in normface
        return output


def get_loss(config: Config):
    loss_type = config.loss_type
    if loss_type == 'ce':
        return ResNetLoss(loss_type='ce')
    if loss_type == 'bce':
        return ResNetLoss(loss_type='bce')
    if loss_type == 'focal':
        return FocalLoss()
    if loss_type == 'am-softmax':
        return Am_softmax().to(config.device)
    if loss_type == 'arcface':
        return None
