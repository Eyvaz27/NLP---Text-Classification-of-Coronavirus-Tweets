from dataclasses import dataclass
from typing import Literal, Optional

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
from .decoder import Decoder

@dataclass
class DecoderMLPCfg:
    name: Literal["mlp"]
    input_size: Optional[int]
    hidden_size: int
    num_layers: int
    output_size: Optional[int]
    non_linearity: Optional[str] 

class DecoderMLP(Decoder[DecoderMLPCfg]):
    model: nn.Module

    def __init__(self, cfg: DecoderMLPCfg) -> None:
        super().__init__(cfg)
        # # I will use ReLU activation
        # # between hidden units 
        self.init_decoder()

    def init_decoder(self):
        self.model = nn.Linear(in_features=self.cfg.input_size, 
                               out_features=self.cfg.output_size)
        layer_in = lambda idx: self.cfg.input_size if idx==0 else self.cfg.hidden_size
        layer_out = lambda idx: self.cfg.output_size if idx==self.cfg.num_layers-1 else self.cfg.hidden_size
        block_layer = lambda features: nn.Sequential(nn.Linear(in_features=features[0], 
                                                               out_features=features[1]), nn.ReLU())
        self.model = nn.Sequential(*[block_layer([layer_in(layer_idx), layer_out(layer_idx)])\
                                     for layer_idx in range(self.cfg.num_layers)])
        if self.cfg.non_linearity == "softmax":
            self.non_linearity = nn.Softmax()
        else: self.non_linearity = nn.Identity()

    def forward(
        self,
        features: Float[Tensor, "batch input_size"]
        ) -> Float[Tensor, "batch output_size"]:

        # # # compute probabilities and return result
        logits = self.model(features)
        return self.non_linearity(logits)
