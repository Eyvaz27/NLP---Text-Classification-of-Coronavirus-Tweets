import torch
import numpy as np
import pandas as pd
from typing import Union, Optional, Literal
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from dataclasses import dataclass
from transformers import BertModel

Stage = Literal["train", "validation", "test"]
TRAIN_FILE_PATH = "/workspaces/NLP---Text-Classification-of-Coronavirus-Tweets/dataset/Corona_NLP_train.csv"
TEST_FILE_PATH = "/workspaces/NLP---Text-Classification-of-Coronavirus-Tweets/dataset/Corona_NLP_test.csv"
CLASS_MAP = {"Extremely Negative": 0.0, "Negative": 1.0, "Neutral": 2.0, "Positive": 3.0, "Extremely Positive": 4.0}


@dataclass
class DatasetCfg:
    max_length: int
    padding: str
    truncation: bool

class TweetDataset(Dataset):
    def __init__(self,
                 cfg: DatasetCfg,  
                 stage: Stage, 
                 random_seed: Optional[int],
                 eval_set_ratio: float=0.2):
        
        assert eval_set_ratio>=0 and eval_set_ratio<=1.0
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