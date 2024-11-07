from torch.utils.data import Dataset
from .dataset_tweet import TweetDataset, TweetDatasetCfg
from .types import Stage
from .view_sampler import get_view_sampler

DATASETS: dict[str, Dataset] = {
    "tweet": TweetDataset}

DatasetCfg = TweetDatasetCfg


def get_dataset(
    cfg: DatasetCfg,
    stage: Stage, seed: int) -> Dataset:
    return DATASETS[cfg.name](cfg, stage, seed)