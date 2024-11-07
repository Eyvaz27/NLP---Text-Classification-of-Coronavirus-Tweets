from dataclasses import dataclass
from typing import Literal, Optional

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
from .decoder import Decoder

@dataclass
class DecoderPerceptronCfg:
    name: Literal["perceptron"]
    input_size: Optional[int]
    output_size: Optional[int]
    non_linearity: Optional[str] 

class DecoderPerceptron(Decoder[DecoderPerceptronCfg]):
    model: nn.Module

    def __init__(self, cfg: DecoderPerceptronCfg) -> None:
        super().__init__(cfg)
        self.init_decoder()

    def init_decoder(self):
        self.model = nn.Linear(in_features=self.cfg.input_size, 
                               out_features=self.cfg.output_size)
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
