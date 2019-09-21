import sys

import numpy as np
from PIL import Image
import torchvision
from torch.utils.data.dataset import Subset


def get_cifar10(root, cfg_trainer, train=True,
                transform_train=None, transform_val=None,
                download=True):
    base_dataset = torchvision.datasets.CIFAR10(root, train=train, download=download)
    train_idxs, val_idxs = train_val_split(base_dataset.targets)
    # num_samples = len(base_dataset)
    # train_size = int(num_samples * 0.9)
    # train_idxs, val_idxs = list(range(0, train_size)), list(range(train_size, num_samples))

    train_dataset = CIFAR10_train(root, cfg_trainer, train=train, transform=transform_train)
    if cfg_trainer['asym']:
        train_dataset.asymmetric_noise()
    else:
        train_dataset.symmetric_noise()
    val_dataset = CIFAR10_val(root, val_idxs, train=train, transform=transform_val)

    print(f"Train: {len(train_idxs)} Val: {len(val_idxs)}")  # Train: 45000 Val: 5000
    return train_dataset, val_dataset


def train_val_split(base_dataset: torchvision.datasets.CIFAR10):
    base_dataset = np.array(base_dataset)
    train_n = int(len(base_dataset) * 0.9 / 10)
    train_idxs = []
    val_idxs = []

    for i in range(10):
        idxs = np.where(base_dataset == i)[0]
        np.random.shuffle(idxs)
        train_idxs.extend(idxs[:train_n])
        val_idxs.extend(idxs[train_n:])
    np.random.shuffle(train_idxs)
    np.random.shuffle(val_idxs)

    return train_idxs, val_idxs


class CIFAR10_train(torchvision.datasets.CIFAR10):
    def __init__(self, root, cfg_trainer, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_train, self).__init__(root, train=train,
                                            transform=transform, target_transform=target_transform,
                                            download=download)
        self.cfg_trainer = cfg_trainer
        self.train_data = self.data
        self.train_labels = np.array(self.targets)
        self.soft_labels = np.zeros((len(self.train_data), 10), dtype=np.float32)
        self.prediction = np.zeros((len(self.train_data), 10, 10), dtype=np.float32)

        self.count = 0

    def symmetric_noise(self):
        indices = np.random.permutation(len(self.train_data))
        for i, idx in enumerate(indices):
            if i < self.cfg_trainer['percent'] * len(self.train_data):
                self.train_labels[idx] = np.random.randint(10, dtype=np.int32)
            self.soft_labels[idx][self.train_labels[idx]] = 1.

    def asymmetric_noise(self):
        for i in range(10):
            indices = np.where(self.train_labels == i)[0]
            np.random.shuffle(indices)
            for j, idx in enumerate(indices):
                if j < self.cfg_trainer['percent'] * len(indices):
                    # truck -> automobile
                    if i == 9:
                        self.train_labels[idx] = 1
                    # bird -> airplane
                    elif i == 2:
                        self.train_labels[idx] = 0
                    # cat -> dog
                    elif i == 3:
                        self.train_labels[idx] = 5
                    # dog -> cat
                    elif i == 5:
                        self.train_labels[idx] = 3
                    # deer -> horse
                    elif i == 4:
                        self.train_labels[idx] = 7
                self.soft_labels[idx][self.train_labels[idx]] = 1.

    def label_update(self, results):
        self.count += 1

        # While updating the noisy label y_i by the probability s,
        # we used the average output probability of the network of the past 10 epochs as s.
        idx = (self.count - 1) % 10
        self.prediction[:, idx] = results

        if self.count >= self.cfg_trainer['begin']:
            self.soft_labels = self.prediction.mean(axis=1)
            self.train_labels = np.argmax(self.soft_labels, axis=1).astype(np.int64)

        if self.count == self.cfg_trainer['epochs']:
            np.save(f'{self.cfg_trainer["save_dir"]}/images.npy', self.train_data)
            np.save(f'{self.cfg_trainer["save_dir"]}/labels.npy', self.train_labels)
            np.save(f'{self.cfg_trainer["save_dir"]}/soft_labels.npy', self.soft_labels)

    def reload_label(self):
        self.train_data = np.load(f'{self.cfg_trainer["label"]}/images.npy')
        self.train_labels = np.load(f'{self.cfg_trainer["label"]}/labels.npy')
        self.soft_labels = np.load(f'{self.cfg_trainer["label"]}/soft_labels.npy')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, soft_target = self.train_data[index], self.train_labels[index], self.soft_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, soft_target, index


class CIFAR10_val(torchvision.datasets.CIFAR10):

    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_val, self).__init__(root, train=train,
                                          transform=transform, target_transform=target_transform,
                                          download=download)

        # self.train_data = self.data[indexs]
        # self.train_labels = np.array(self.targets)[indexs]
        self.train_data = self.data
        self.train_labels = np.array(self.targets)
