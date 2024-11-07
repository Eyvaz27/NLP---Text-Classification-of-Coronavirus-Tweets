import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass
from transformers import BertModel
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Union, Optional, Literal
from .meta_info import Stage, TRAIN_FILE_PATH, TEST_FILE_PATH, CLASS_MAP

@dataclass
class TweetDatasetCfg:
    name: Literal["tweet"]
    max_length: int
    padding: str
    truncation: bool
    eval_set_ratio: float

class TweetDataset(Dataset):
    def __init__(self,
                 cfg: TweetDatasetCfg,  
                 stage: Stage, 
                 random_seed: Optional[int]):
        
        self.eval_set_ratio = self.cfg.eval_set_ratio
        assert self.eval_set_ratio>=0 and self.eval_set_ratio<=1.0

        self.seed = random_seed if random_seed else 42
        self.stage = stage
        self.cfg = cfg

        # read data from memory
        self.load_csv_file()
        self.init_tokenizer()

    def load_csv_file(self):
        if self.stage == "test":
            raw_csv_file = pd.read_csv(TEST_FILE_PATH, encoding = 'latin1')
        else: raw_csv_file = pd.read_csv(TRAIN_FILE_PATH, encoding = 'latin1')
        
        tweet_lower = lambda tweet: tweet.lower()
        feature_texts = raw_csv_file["OriginalTweet"].map(tweet_lower)
        label_texts = raw_csv_file["Sentiment"].map(CLASS_MAP)

        if self.stage == "test":
            self.feature_texts = feature_texts
            self.label_texts = label_texts
        else:
            np.random.seed(seed=self.seed)
            random_perm_indexes = np.random.permutation(len(feature_texts))
            train_sample_indexes = random_perm_indexes[:int((1.0 - self.eval_set_ratio) * len(random_perm_indexes))]
            validation_sample_indexes = random_perm_indexes[int((1.0 - self.eval_set_ratio) * len(random_perm_indexes)):]
            if self.stage == "train":
                self.feature_texts = feature_texts[train_sample_indexes]
                self.label_texts = label_texts[train_sample_indexes]
            elif self.stage == "validation":
                self.feature_texts = feature_texts[validation_sample_indexes]
                self.label_texts = label_texts[validation_sample_indexes]
    
    def init_tokenizer(self):
        self.auto_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        self.tokenizer = lambda tweet: self.auto_tokenizer(tweet, padding=self.cfg.padding, truncation=self.cfg.truncation,
                                                           max_length=self.cfg.max_length, return_tensors="pt")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased", torch_dtype=torch.float16, attn_implementation="sdpa")
        self.embedding_layer = lambda tweet_tokens: self.bert_model.embeddings(tweet_tokens['input_ids'])
        
    def __len__(self):
        return len(self.feature_texts)

    def __getitem__(self, idx):
        # # # 
        # # # retrieving samples from the dataset
        sample_tweet = self.feature_texts[idx]
        sample_label = self.label_texts[idx]
        return self.embedding_layer(self.tokenizer(sample_tweet)), sample_label