from typing import Literal

Stage = Literal["train", "validation", "test"]
TRAIN_FILE_PATH = "/workspaces/NLP---Text-Classification-of-Coronavirus-Tweets/dataset/Corona_NLP_train.csv"
TEST_FILE_PATH = "/workspaces/NLP---Text-Classification-of-Coronavirus-Tweets/dataset/Corona_NLP_test.csv"
CLASS_MAP = {"Extremely Negative": 0, "Negative": 1, "Neutral": 2, "Positive": 3, "Extremely Positive": 4}