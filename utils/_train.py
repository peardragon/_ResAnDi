import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import pkbar

class EarlyStopping:

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): validation loss 가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): Default: False
            delta (float): monitored quantity 의 최소 변화
            path (str): Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)

        # -val_loss 가 클 수록 더 나은 모델,
        # score < best_score ( -val_loss ) 이면, 모델이 더 나아지지 않음. count.
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            print(f'Validation loss  ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        validation loss 가 감소하면 모델을 저장한다.
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def train(model, num_epochs, train_dataloaders, eval_dataloader, device="cuda",
          save_path="./model_ckpt/", lr=0.001, step_size=10, gamma=0.5, patience=10, optimizer=None, log_dir=None):
    # DEFAULT SETTINGS #
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), **dict(lr=lr))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **dict(step_size=step_size, gamma=gamma))
    loss_func = torch.nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=patience, verbose=True, delta=0.001, path=save_path+"checkpoint.pt")
    tb = SummaryWriter(comment='', log_dir=log_dir)
    ####################

    model = model.to(device)
    for epoch in range(num_epochs):

        # progress bar
        kbar = pkbar.Kbar(target=len(train_dataloaders), epoch=epoch,
                          num_epochs=num_epochs, width=50, always_stateful=True)

        train_loss_total = 0.0
        train_acc_total = 0.0
        model.train()
        for i, train_dataloader in enumerate(train_dataloaders):
            loss_total_dataloader = 0.0
            acc_total_dataloader = 0.0

            for dataset in train_dataloader:
                (X,y) = dataset
                inputs = X.to(device)
                target = y.to(device)

                optimizer.zero_grad()
                results = model(inputs)
                loss = loss_func(results, target)
                loss_total_dataloader += loss.item()

                loss.backward()
                optimizer.step()

                acc = torch.sum(torch.Tensor(results.argmax(axis=-1) == target))/target.size()[0]
                acc_total_dataloader += acc.item()

            loss_avg = loss_total_dataloader / len(train_dataloader)
            acc_avg = acc_total_dataloader / len(train_dataloader)
            train_loss_total += loss_avg
            train_acc_total += acc_avg

            kbar.update(i+1, values=[("loss", loss_avg), ("acc", acc_avg)])

        loss_total_avg = train_loss_total/len(train_dataloaders)
        torch.save(model.state_dict(), save_path + f"ckpt_{epoch}.pt")
        tb.add_scalar("Train Loss (avg)", loss_total_avg, epoch)

        eval_loss_total = 0.0
        eval_acc_total = 0.0
        model.eval()
        with torch.no_grad():
            for dataset in eval_dataloader:
                (X,y) = dataset
                inputs = X.to(device)
                target = y.to(device)

                results = model(inputs)
                loss = loss_func(results, target)
                eval_loss_total += loss.item()

                acc = torch.sum(torch.Tensor(results.argmax(axis=-1) == target))/target.size()[0]
                eval_acc_total += acc.item()

        eval_loss_avg = eval_loss_total / len(eval_dataloader)
        eval_acc_avg = eval_acc_total / len(eval_dataloader)

        tb.add_scalar("Val Loss (avg)", eval_loss_avg, epoch)
        tb.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], epoch)
        scheduler.step()

        early_stopping(eval_loss_avg, model)
        print("Validation Accuracy :", eval_acc_avg)

        if early_stopping.early_stop:
            print("Early Stopping")
            break

    tb.close()
