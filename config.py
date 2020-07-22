import torch


class Config():
    N_CLASS = 264
    IMG_WIDTH = 224
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 16
    NUM_WORKER = 16
    DATA_PATH = './data/'
    model_name = "resnet50"  # efficientnet-b0
    N_EPOCH = 50
    expriment_id = 1
    split_n = 5
    use_half = False
    loss_type = 'bce' # focal
    lr = 0.001
    scheduler_type = 'Plateau'  # cos
    swa = False
