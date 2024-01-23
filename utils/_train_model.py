from utils._resnet import resnet18_8
import os
import numpy as np
from utils._preprocessing import _preprocessing
from utils._train import train
import shutil


def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def _train():
    # DATASET Preprocessing parameter setting #

    train_dir_size = 40
    BATCH_SIZE = 64
    lr = 0.0001
    shuffle = True
    increment = False
    scale = "minmax"
    # scale = "simple"
    fpath = f"./backups/resnet18_8_b{BATCH_SIZE}_lr{lr}/"
    model = resnet18_8()

    #############################################

    # EVAL- DATASET PREPARE #
    print("Eval dataset prepare")

    eval_set = np.load("./dataset/db/eval/0.npy", allow_pickle=True)
    eval_dataloader = _preprocessing(eval_set, batch_size=BATCH_SIZE)
    print("DONE")

    # TRAIN- DATASET PREPARE #
    print("Train dataset prepare")

    train_dataloaders = []

    for i in range(train_dir_size):
        print(f"{i / train_dir_size * 100:.2f}%..", end="\r")
        train_set = np.load(f"./dataset/db/train/{i}.npy", allow_pickle=True)
        train_dataloader = _preprocessing(train_set, batch_size=BATCH_SIZE)
        train_dataloaders.append(train_dataloader)

    print("DONE")

    ###########################
    train(model, num_epochs=100, train_dataloaders=train_dataloaders, eval_dataloader=eval_dataloader,
          save_path="./model_ckpt/", lr=lr)

    create_folder(fpath)
    shutil.copy("./model_ckpt/checkpoint.pt", fpath + "checkpoint.pt")


def _train_1000():
    train_dir_size = 2
    BATCH_SIZE = 256
    lr = 0.001
    shuffle = True
    increment = False
    scale = "minmax"
    # scale = "simple"
    fpath = f"./model_backups/resnet18_8_b{BATCH_SIZE}_lr{lr}_1000/"
    model = resnet18_8()

    #############################################

    # EVAL- DATASET PREPARE #
    print("Eval dataset prepare")

    eval_set = np.load("./dataset/eval_1000/0.npy", allow_pickle=True)
    eval_dataloader = _preprocessing(eval_set, batch_size=BATCH_SIZE)
    print("DONE")

    # TRAIN- DATASET PREPARE #
    print("Train dataset prepare")

    train_dataloaders = []

    for i in range(train_dir_size):
        print(f"{i / train_dir_size * 100:.2f}%..", end="\r")
        train_set = np.load(f"./dataset/train_1000/{i}.npy", allow_pickle=True)
        train_dataloader = _preprocessing(train_set, batch_size=BATCH_SIZE)
        train_dataloaders.append(train_dataloader)

    print("DONE")

    ###########################
    train(model, num_epochs=100, train_dataloaders=train_dataloaders, eval_dataloader=eval_dataloader,
          save_path="./model_ckpt/", lr=lr)

    create_folder(fpath)
    shutil.copy("./model_ckpt/checkpoint.pt", fpath + "checkpoint.pt")


def aug_train(feat, opt, p, tag):
    # DATASET Preprocessing parameter setting #

    train_dir_size = 1
    BATCH_SIZE = 64
    lr = 0.0001
    model_tag_name = f"resnet18_8_b{BATCH_SIZE}_lr{lr}_1000_augmentation_{feat}_{opt}_{p}"

    fpath = f"./model_backups/{model_tag_name}/"
    model = resnet18_8()

    if os.path.exists(fpath + f"checkpoint{tag}.pt"):
        print(fpath, " Exist, skip")
        return

    #############################################

    # EVAL- DATASET PREPARE #
    print("Eval dataset prepare")

    eval_set = np.load("./dataset/eval_1000/0.npy", allow_pickle=True)
    eval_dataloader = _preprocessing(eval_set, batch_size=BATCH_SIZE)
    print("DONE")

    # TRAIN- DATASET PREPARE #
    print("Train dataset prepare")

    train_dataloaders = []
    for i in range(train_dir_size):
        print(f"{i / train_dir_size * 100:.2f}%..", end="\r")
        train_set = np.load(f"./datast/train_1000/{i}_p{p}.npy", allow_pickle=True)
        train_dataloader = _preprocessing(train_set, batch_size=BATCH_SIZE)
        train_dataloaders.append(train_dataloader)

    print("DONE")

    ###########################
    train(model, num_epochs=5, train_dataloaders=train_dataloaders, eval_dataloader=eval_dataloader,
          save_path="./model_ckpt/", lr=lr, patience=5, log_dir=f"./runs/{model_tag_name}")

    create_folder(fpath)
    shutil.copy("./model_ckpt/checkpoint.pt", fpath + f"checkpoint{tag}.pt")


def aug_train_iter():
    for i in range(20):
        for p in [50, 60, 70, 80, 90, 100]:
            aug_train("gc", "mean", p, i)

