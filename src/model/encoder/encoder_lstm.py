from dataclasses import dataclass
from typing import Literal, Optional

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
from .encoder import Encoder

@dataclass
class EncoderLSTMCfg:
    name: Literal["lstm"]
    input_size: Optional[int]
    hidden_size: int
    num_layers: int
    bias: bool
    batch_first: bool
    dropout: float
    bidirectional: bool
    aggregation: str

class EncoderLSTM(Encoder[EncoderLSTMCfg]):
    def __init__(self, cfg: EncoderLSTMCfg) -> None:
        super().__init__(cfg)
        
    def reset_hidden_cell(self, seq_length):
        D = 2 if self.cfg.bidirectional else 1
        if self.cfg.batch_first:
            h0 = torch.randn(D*self.cfg.num_layers, seq_length, self.cfg.hidden_size)
            c0 = torch.randn(D*self.cfg.num_layers, seq_length, self.cfg.hidden_size)
        else:
            h0 = torch.randn(seq_length, D*self.cfg.num_layers, self.cfg.hidden_size)
            c0 = torch.randn(seq_length, D*self.cfg.num_layers, self.cfg.hidden_size)
        return h0, c0
    
    def init_encoder(self):
        self.model = nn.LSTM(input_size=self.cfg.input_size, hidden_size=self.cfg.hidden_size, 
                             num_layers=self.cfg.num_layers, bias=self.cfg.bias, 
                             batch_first=self.cfg.batch_first, dropout=self.cfg.dropout,
                             bidirectional=self.cfg.bidirectional)

    def forward(
        self,
        features: Float[Tensor, "batch seq dim"]
        ) -> Float[Tensor, "batch"]:
        
        _, s, _ = features.shape
        # # # initialize hidden and cell states
        h0, c0 = self.reset_hidden_cell(seq_length=s)
        # # # run LSTM model on token embeddings
        output, (hn, cn) = self.model(features, (h0, c0))

        first_token = output[:, 0, :]
        last_token = output[:, -1, :]
        if self.cfg.aggregation == "sum":
            return first_token + last_token
        elif self.cfg.aggregation == "concat":
            return torch.concatenate([first_token, last_token], axis=-1)
        else:
            raise KeyError("Required aggregation is not implemented yet ...")