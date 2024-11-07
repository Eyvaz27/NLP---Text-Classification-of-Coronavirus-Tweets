import random
from dataclasses import dataclass
from typing import Callable

import torch
import numpy as np
from torch import Generator, nn
from torch.utils.data import DataLoader, Dataset

from . import DatasetCfg, get_dataset
from .meta_info import Stage

@dataclass
class DataLoaderStageCfg:
    batch_size: int
    num_workers: int
    persistent_workers: bool
    random_seed: int | None


@dataclass
class DataLoaderCfg:
    train: DataLoaderStageCfg
    test: DataLoaderStageCfg
    val: DataLoaderStageCfg

def worker_init_fn(worker_id: int) -> None:
    random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))
    np.random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))

class DataModule(LightningDataModule):
    dataset_cfg: DatasetCfg
    data_loader_cfg: DataLoaderCfg
    global_rank: int

    def __init__(
        self,
        dataset_cfg: DatasetCfg,
        data_loader_cfg: DataLoaderCfg,
        global_rank: int = 0,
    ) -> None:
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.data_loader_cfg = data_loader_cfg
        self.global_rank = global_rank

    def get_persistent(self, loader_cfg: DataLoaderStageCfg) -> bool | None:
        return None if loader_cfg.num_workers == 0 else loader_cfg.persistent_workers

    def get_generator(self, loader_cfg: DataLoaderStageCfg) -> torch.Generator | None:
        if loader_cfg.seed is None:
            return None
        generator = Generator()
        generator.manual_seed(loader_cfg.seed + self.global_rank)
        return generator

    def train_dataloader(self):
        dataset = get_dataset(self.dataset_cfg, "train", 
                                self.data_loader_cfg.random_seed)
        return DataLoader(
            dataset,
            self.data_loader_cfg.train.batch_size,
            shuffle=True,num_workers=self.data_loader_cfg.train.num_workers,
            generator=self.get_generator(self.data_loader_cfg.train),
            persistent_workers=self.get_persistent(self.data_loader_cfg.train),
            worker_init_fn=worker_init_fn)

    def val_dataloader(self):
        dataset = get_dataset(self.dataset_cfg, "val", 
                                self.data_loader_cfg.random_seed)
        return DataLoader(
            dataset,
            self.data_loader_cfg.val.batch_size,
            num_workers=self.data_loader_cfg.val.num_workers,
            generator=self.get_generator(self.data_loader_cfg.val),
            persistent_workers=self.get_persistent(self.data_loader_cfg.val),
            worker_init_fn=worker_init_fn)

    def test_dataloader(self):
        dataset = get_dataset(self.dataset_cfg, "test", 
                                self.data_loader_cfg.random_seed)
        return DataLoader(
            dataset,
            self.data_loader_cfg.test.batch_size,
            num_workers=self.data_loader_cfg.test.num_workers,
            generator=self.get_generator(self.data_loader_cfg.test),
            persistent_workers=self.get_persistent(self.data_loader_cfg.test),
            worker_init_fn=worker_init_fn)