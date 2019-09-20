import numpy as np
import torch
from tqdm import tqdm
from typing import List
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, train_criterion, metrics, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None, val_criterion=None):
        super().__init__(model, train_criterion, metrics, optimizer, config, val_criterion)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.train_loss_list: List[float] = []
        self.val_loss_list: List[float] = []

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        results = np.zeros((len(self.data_loader.dataset), 10), dtype=np.float32)
        with tqdm(self.data_loader) as progress:
            for batch_idx, (data, target, soft_targets, indexs) in enumerate(progress):
                progress.set_description_str(f'Train epoch {epoch}')
                data, target = data.to(self.device), target.to(self.device)
                soft_targets, indexs = soft_targets.to(self.device), indexs.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                probs, loss = self.train_criterion(output, soft_targets)

                results[indexs.cpu().detach().numpy().tolist()] = probs.cpu().detach().numpy().tolist()  # 要確認？

                loss.backward()
                self.optimizer.step()

                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.writer.add_scalar('loss', loss.item())
                self.train_loss_list.append(loss.item())
                total_loss += loss.item()
                total_metrics += self._eval_metrics(output, target)

                if batch_idx % self.log_step == 0:
                    progress.set_postfix_str(' {} Loss: {:.6f}'.format(
                        self._progress(batch_idx),
                        loss.item()))
                    # self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    #     epoch,
                    #     self._progress(batch_idx),
                    #     loss.item()))
                    self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

                if batch_idx == self.len_epoch:
                    break

        log = {
            'loss': total_loss / self.len_epoch,
            'metrics': (total_metrics / self.len_epoch).tolist()
        }

        # update soft labels （論文には、CIFAR10の場合、ラベルの更新は70epoch目からとあるが...?）
        if epoch > 69:
            self.data_loader.dataset.label_update(results)

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(val_log)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            with tqdm(self.valid_data_loader) as progress:
                for batch_idx, (data, target) in enumerate(progress):
                    progress.set_description_str(f'Valid epoch {epoch}')
                    data, target = data.to(self.device), target.to(self.device)

                    output = self.model(data)
                    loss = self.val_criterion(output, target)

                    self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                    self.writer.add_scalar('loss', loss.item())
                    self.val_loss_list.append(loss.item())
                    total_val_loss += loss.item()
                    total_val_metrics += self._eval_metrics(output, target)
                    self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
